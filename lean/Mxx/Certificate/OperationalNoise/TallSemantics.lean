import Mxx.Certificate.OperationalNoise.TallSecurity0ABI

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSemantics

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1
open TallSecurity0ABI

/-! Pure semantic statements shared by the fixed toy and Tall certificate ABIs. -/

/-- A semantic environment assigns values only to factors that occur in monomial keys. -/
abbrev Env (Factor : Type) := Factor → Int

/-! Tall factors are not assigned an artificial numeric order. Their central representation is
    the source order; semantic equalities use `List.Perm` below. -/
instance ownerCentralNormalizer : CentralNormalizer Owner where
  normalize := id

def evalMonomial {Factor : Type} (env : Env Factor) (key : MonomialKey Factor) : Int :=
  (key.centralFactors.map env).prod * (key.orderedFactors.map env).prod

def evalPolynomial {Factor : Type} (env : Env Factor) : Polynomial Factor → Int
  | [] => 0
  | term :: terms =>
      term.coefficient * evalMonomial env term.key + evalPolynomial env terms

def termPolynomial (terms : List Term) : Polynomial Owner := terms.map Term.toExact

def replayProductFacts (facts : TallSecurity0ABI.ProductFacts) : EventReplay.ProductFacts :=
  { leftConstantPolynomial := facts.leftIsConstantPolynomial
    rightConstantPolynomial := facts.rightIsConstantPolynomial
    rightKnownZeroRows := facts.rightKnownZeroRows
    leftSupportUpper := facts.leftSupportUpper
    rightSupportUpper := facts.rightSupportUpper }

def relationContext (source : MonomialKey Owner)
    (exteriorCentral : List Owner)
    (orderedStart orderedEndExclusive : Nat) :
  MonomialContext Owner :=
  { exteriorCentral := exteriorCentral
    prefixFactors := source.orderedFactors.take orderedStart
    suffixFactors := source.orderedFactors.drop orderedEndExclusive }

private def extractFirst (x : Owner) : List Owner → Option (List Owner)
  | [] => none
  | y :: ys =>
      if x = y then some ys
      else
        match extractFirst x ys with
        | none => none
        | some rest => some (y :: rest)

private theorem extractFirst_sound {x : Owner} {ys rest : List Owner}
    (h : extractFirst x ys = some rest) : ys.Perm (x :: rest) := by
  induction ys generalizing rest with
  | nil => simp [extractFirst] at h
  | cons y ys ih =>
      by_cases eq : x = y
      · simp [extractFirst, eq] at h
        cases eq
        cases h
        exact List.Perm.refl _
      · simp only [extractFirst, eq, ↓reduceIte] at h
        cases found : extractFirst x ys with
        | none => simp [found] at h
        | some tail =>
            simp only [found, Option.some.injEq] at h
            subst rest
            have tailPerm : ys.Perm (x :: tail) := ih found
            exact (List.Perm.cons y tailPerm).trans (List.Perm.swap x y tail)

private theorem extractFirst_exists {x : Owner} {ys : List Owner}
    (h : x ∈ ys) : ∃ rest, extractFirst x ys = some rest := by
  induction ys with
  | nil => simp at h
  | cons y ys ih =>
      by_cases eq : x = y
      · subst y
        exact ⟨ys, by simp [extractFirst]⟩
      · simp only [List.mem_cons, eq, false_or] at h
        rcases ih h with ⟨rest, found⟩
        exact ⟨y :: rest, by simp [extractFirst, eq, found]⟩

private theorem extractFirst_complete {x : Owner} {ys rest : List Owner}
    (h : ys.Perm (x :: rest)) :
    ∃ extracted, extractFirst x ys = some extracted ∧ extracted.Perm rest := by
  have present : x ∈ ys := h.mem_iff.mpr (by simp)
  rcases extractFirst_exists present with ⟨extracted, found⟩
  refine ⟨extracted, found, ?_⟩
  exact (List.Perm.cons_inv (h.symm.trans (extractFirst_sound found))).symm

private def permCheck : List Owner → List Owner → Bool
  | [], ys => ys == []
  | x :: xs, ys =>
      match extractFirst x ys with
      | none => false
      | some rest => permCheck xs rest

private theorem permCheck_iff {left right : List Owner} :
    permCheck left right = true ↔ left.Perm right := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => simp [permCheck]
      | cons head tail =>
          constructor
          · simp [permCheck]
          · intro h
            exact False.elim (by
              have impossible := h.symm.eq_nil
              cases impossible)
  | cons head tail ih =>
      cases right with
      | nil =>
          constructor
          · simp [permCheck, extractFirst]
          · intro h
            exact False.elim (by
              have impossible := h.eq_nil
              cases impossible)
      | cons rightHead rightTail =>
          cases found : extractFirst head (rightHead :: rightTail) with
          | none =>
              constructor
              · simp [permCheck, found]
              · intro h
                exact False.elim (by
                  rcases extractFirst_complete h.symm with ⟨rest, found', _⟩
                  rw [found] at found'
                  cases found')
          | some extracted =>
              constructor
              · intro checked
                simp only [permCheck, found] at checked
                have tailPerm : tail.Perm extracted := (ih (right := extracted)).mp checked
                have extractedPerm : (rightHead :: rightTail).Perm (head :: extracted) :=
                  extractFirst_sound found
                exact (List.Perm.cons head tailPerm).trans extractedPerm.symm
              · intro permutation
                simp only [permCheck, found]
                have extractedPerm : (rightHead :: rightTail).Perm (head :: extracted) :=
                  extractFirst_sound found
                have tailPerm : tail.Perm extracted :=
                  List.Perm.cons_inv (permutation.trans extractedPerm)
                exact (ih (right := extracted)).mpr tailPerm

def KeyEquivalent (left right : MonomialKey Owner) : Prop :=
  left.centralFactors.Perm right.centralFactors ∧
    left.orderedFactors = right.orderedFactors

/-! Canonical comparison is local to this semantic layer.  In particular, it does not encode an
    `Owner` as a `Nat`: scope tags and their contained rows are compared first, followed by the
    expression row.  This gives central-factor sorting a total structural order without adding a
    premise to the Rust certificate. -/

private def compareNat (left right : Nat) : Ordering :=
  if left < right then .lt else if right < left then .gt else .eq

def ownerCompare (left right : Owner) : Ordering :=
  let scopeComparison := match left.scope, right.scope with
    | .closed leftRoot, .closed rightRoot => compareNat leftRoot.row rightRoot.row
    | .closed _, .program _ => .lt
    | .program _, .closed _ => .gt
    | .program leftProgram, .program rightProgram =>
        compareNat leftProgram.row rightProgram.row
  match scopeComparison with
  | .eq => compareNat left.expression.row right.expression.row
  | comparison => comparison

def ownerLe (left right : Owner) : Bool := (ownerCompare left right).isLE

private def keyCompare (left right : MonomialKey Owner) : Ordering :=
  match List.compareLex ownerCompare left.centralFactors right.centralFactors with
  | .eq => List.compareLex ownerCompare left.orderedFactors right.orderedFactors
  | comparison => comparison

private def termCompare (left right : ExactTerm Owner) : Ordering :=
  keyCompare left.key right.key

private def termLe (left right : ExactTerm Owner) : Bool := (termCompare left right).isLE

private def constructiveMergeAux {α : Type} (le : α → α → Bool) :
    List α → List α → Nat → List α
  | [], right, _ => right
  | left, [], _ => left
  | leftHead :: leftTail, rightHead :: rightTail, 0 => leftHead :: leftTail ++ rightHead :: rightTail
  | leftHead :: leftTail, rightHead :: rightTail, fuel + 1 =>
      if le leftHead rightHead then
        leftHead :: constructiveMergeAux le leftTail (rightHead :: rightTail) fuel
      else
        rightHead :: constructiveMergeAux le (leftHead :: leftTail) rightTail fuel

private def constructiveMerge {α : Type} (le : α → α → Bool)
    (left right : List α) : List α :=
  match right with
  | [] => left
  | rightHead :: rightTail =>
      match left with
      | [] => rightHead :: rightTail
      | leftHead :: leftTail =>
          constructiveMergeAux le (leftHead :: leftTail) (rightHead :: rightTail)
            ((leftHead :: leftTail).length + (rightHead :: rightTail).length)

private theorem constructiveMerge_nil_left {α : Type} (le : α → α → Bool)
    (right : List α) : constructiveMerge le [] right = right := by
  cases right <;> rfl

private theorem constructiveMerge_nil_right {α : Type} (le : α → α → Bool)
    (left : List α) : constructiveMerge le left [] = left := by
  unfold constructiveMerge
  rfl

private theorem constructiveMerge_cons_cons {α : Type} (le : α → α → Bool)
    (leftHead rightHead : α) (leftTail rightTail : List α) :
    constructiveMerge le (leftHead :: leftTail) (rightHead :: rightTail) =
      if le leftHead rightHead then
        leftHead :: constructiveMerge le leftTail (rightHead :: rightTail)
      else
        rightHead :: constructiveMerge le (leftHead :: leftTail) rightTail := by
  unfold constructiveMerge
  cases leftTail <;> cases rightTail <;>
    simp [constructiveMergeAux, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]

private def constructiveMergePassAux {α : Type} (le : α → α → Bool) (width : Nat) :
    Nat → List α → List α
  | _fuel, [] => []
  | fuel, factors =>
      if width = 0 then factors
      else match fuel with
        | 0 => factors
        | fuel + 1 =>
            let left := factors.take width
            let remainder := factors.drop width
            let right := remainder.take width
            let tail := remainder.drop width
            constructiveMerge le left right ++
              constructiveMergePassAux le width fuel tail

private def constructiveMergePass {α : Type} (le : α → α → Bool)
    (width : Nat) (factors : List α) : List α :=
  constructiveMergePassAux le width factors.length factors

private def constructiveMergeSortAux {α : Type} (le : α → α → Bool)
    (factors : List α) (width : Nat) : Nat → List α
  | 0 => factors
  | fuel + 1 =>
      if width < factors.length then
        constructiveMergeSortAux le (constructiveMergePass le width factors) (width * 2) fuel
      else factors

private def constructiveMergeSort {α : Type} (le : α → α → Bool) (factors : List α) :
    List α :=
  constructiveMergeSortAux le factors 1 factors.length

private theorem constructiveMerge_prod {α : Type} (env : Env α)
    (le : α → α → Bool) (left right : List α) :
    ((constructiveMerge le left right).map env).prod =
      (left.map env).prod * (right.map env).prod := by
  let motive : Nat → Prop := fun total =>
    ∀ (left right : List α), left.length + right.length = total →
      ((constructiveMerge le left right).map env).prod =
        (left.map env).prod * (right.map env).prod
  refine Nat.strongRecOn (motive := motive) (left.length + right.length) ?_ left right rfl
  intro total inductionHypothesis left right totalEq
  cases left with
  | nil => simp [constructiveMerge_nil_left]
  | cons leftHead leftTail =>
      cases right with
      | nil => simp [constructiveMerge_nil_right]
      | cons rightHead rightTail =>
          cases hb : le leftHead rightHead with
          | false =>
            rw [constructiveMerge_cons_cons]
            simp only [hb, Bool.false_eq_true, ↓reduceIte]
            simp only [List.map, List.prod_cons]
            have recursive := inductionHypothesis
              ((leftHead :: leftTail).length + rightTail.length)
              (by rw [← totalEq]; simp) (leftHead :: leftTail) rightTail rfl
            rw [recursive]
            simp [Int.mul_assoc, Int.mul_left_comm]
          | true =>
            rw [constructiveMerge_cons_cons]
            simp only [hb, ↓reduceIte]
            simp only [List.map, List.prod_cons]
            have recursive := inductionHypothesis
              (leftTail.length + (rightHead :: rightTail).length)
              (by rw [← totalEq]; simp) leftTail (rightHead :: rightTail) rfl
            rw [recursive]
            simp [Int.mul_assoc, Int.mul_left_comm]

private theorem constructiveMergePassAux_prod {α : Type} (env : Env α)
    (le : α → α → Bool) (width fuel : Nat) (factors : List α) :
    ((constructiveMergePassAux le width fuel factors).map env).prod =
      (factors.map env).prod := by
  induction fuel generalizing factors with
  | zero => cases factors <;> simp [constructiveMergePassAux]
  | succ fuel ih =>
      cases factors with
      | nil => simp [constructiveMergePassAux]
      | cons head tail =>
          by_cases widthZero : width = 0
          · simp [constructiveMergePassAux, widthZero]
          · simp only [constructiveMergePassAux, widthZero, ↓reduceIte]
            let left := (head :: tail).take width
            let remainder := (head :: tail).drop width
            let right := remainder.take width
            let rest := remainder.drop width
            have rightRest : right ++ rest = remainder := by
              exact List.take_append_drop width remainder
            have leftRemainder : left ++ remainder = head :: tail := by
              exact List.take_append_drop width (head :: tail)
            have leftRightRest : left ++ right ++ rest = head :: tail := by
              rw [List.append_assoc, rightRest, leftRemainder]
            rw [List.map_append, List.prod_append, constructiveMerge_prod env le left right,
              ih rest]
            rw [← List.prod_append, ← List.prod_append, ← List.map_append, ← List.map_append,
              leftRightRest]

private theorem constructiveMergePass_prod {α : Type} (env : Env α)
    (le : α → α → Bool) (width : Nat) (factors : List α) :
    ((constructiveMergePass le width factors).map env).prod =
      (factors.map env).prod := by
  exact constructiveMergePassAux_prod env le width factors.length factors

private theorem constructiveMergeSortAux_prod {α : Type} (env : Env α)
    (le : α → α → Bool) (factors : List α) (width fuel : Nat) :
    ((constructiveMergeSortAux le factors width fuel).map env).prod =
      (factors.map env).prod := by
  induction fuel generalizing factors width with
  | zero => rfl
  | succ fuel ih =>
      by_cases widthLt : width < factors.length
      · simp only [constructiveMergeSortAux, widthLt, ↓reduceIte]
        rw [ih, constructiveMergePass_prod]
      · simp only [constructiveMergeSortAux, widthLt, ↓reduceIte]

private theorem constructiveMergeSort_prod {α : Type} (env : Env α)
    (le : α → α → Bool) (factors : List α) :
    ((constructiveMergeSort le factors).map env).prod =
      (factors.map env).prod := by
  exact constructiveMergeSortAux_prod env le factors 1 factors.length

private theorem evalPolynomial_append_forCanonical (env : Env Owner)
    (left right : Polynomial Owner) :
    evalPolynomial env (left ++ right) = evalPolynomial env left + evalPolynomial env right := by
  induction left with
  | nil => simp [evalPolynomial]
  | cons term left ih =>
      simp only [List.cons_append, evalPolynomial]
      rw [ih]
      simp [Int.add_assoc]

private theorem constructiveMerge_evalPolynomial (env : Env Owner)
    (le : ExactTerm Owner → ExactTerm Owner → Bool) (left right : Polynomial Owner) :
    evalPolynomial env (constructiveMerge le left right) =
      evalPolynomial env left + evalPolynomial env right := by
  let motive : Nat → Prop := fun total =>
    ∀ (left right : Polynomial Owner), left.length + right.length = total →
      evalPolynomial env (constructiveMerge le left right) =
        evalPolynomial env left + evalPolynomial env right
  refine Nat.strongRecOn (motive := motive) (left.length + right.length) ?_ left right rfl
  intro total inductionHypothesis left right totalEq
  cases left with
  | nil => simp [constructiveMerge_nil_left, evalPolynomial]
  | cons leftHead leftTail =>
      cases right with
      | nil => simp [constructiveMerge_nil_right, evalPolynomial]
      | cons rightHead rightTail =>
          cases hb : le leftHead rightHead with
          | false =>
            rw [constructiveMerge_cons_cons]
            simp only [hb, Bool.false_eq_true, ↓reduceIte]
            simp only [evalPolynomial]
            have recursive := inductionHypothesis
              ((leftHead :: leftTail).length + rightTail.length)
              (by rw [← totalEq]; simp) (leftHead :: leftTail) rightTail rfl
            change rightHead.coefficient * evalMonomial env rightHead.key +
              evalPolynomial env (constructiveMerge le (leftHead :: leftTail) rightTail) = _
            rw [recursive]
            simp only [evalPolynomial]
            simp [Int.add_assoc, Int.add_left_comm]
          | true =>
            rw [constructiveMerge_cons_cons]
            simp only [hb, ↓reduceIte]
            simp only [evalPolynomial]
            have recursive := inductionHypothesis
              (leftTail.length + (rightHead :: rightTail).length)
              (by rw [← totalEq]; simp) leftTail (rightHead :: rightTail) rfl
            change leftHead.coefficient * evalMonomial env leftHead.key +
              evalPolynomial env (constructiveMerge le leftTail (rightHead :: rightTail)) = _
            rw [recursive]
            simp only [evalPolynomial]
            simp [Int.add_assoc, Int.add_left_comm]

private theorem constructiveMergePassAux_evalPolynomial (env : Env Owner)
    (le : ExactTerm Owner → ExactTerm Owner → Bool)
    (width fuel : Nat) (factors : Polynomial Owner) :
    evalPolynomial env (constructiveMergePassAux le width fuel factors) =
      evalPolynomial env factors := by
  induction fuel generalizing factors with
  | zero => cases factors <;> simp [constructiveMergePassAux]
  | succ fuel ih =>
      cases factors with
      | nil => simp [constructiveMergePassAux, evalPolynomial]
      | cons head tail =>
          by_cases widthZero : width = 0
          · simp [constructiveMergePassAux, widthZero]
          · simp only [constructiveMergePassAux, widthZero, ↓reduceIte]
            let left := (head :: tail).take width
            let remainder := (head :: tail).drop width
            let right := remainder.take width
            let rest := remainder.drop width
            have rightRest : right ++ rest = remainder := by
              exact List.take_append_drop width remainder
            have leftRemainder : left ++ remainder = head :: tail := by
              exact List.take_append_drop width (head :: tail)
            have leftRightRest : left ++ right ++ rest = head :: tail := by
              rw [List.append_assoc, rightRest, leftRemainder]
            have merged := constructiveMerge_evalPolynomial env le left right
            have restEval := ih rest
            rw [evalPolynomial_append_forCanonical, merged, restEval]
            rw [← evalPolynomial_append_forCanonical env left right,
              ← evalPolynomial_append_forCanonical env (left ++ right) rest,
              leftRightRest]

private theorem constructiveMergePass_evalPolynomial (env : Env Owner)
    (le : ExactTerm Owner → ExactTerm Owner → Bool)
    (width : Nat) (factors : Polynomial Owner) :
    evalPolynomial env (constructiveMergePass le width factors) =
      evalPolynomial env factors := by
  exact constructiveMergePassAux_evalPolynomial env le width factors.length factors

private theorem constructiveMergeSort_evalPolynomialAux (env : Env Owner)
    (le : ExactTerm Owner → ExactTerm Owner → Bool)
    (factors : Polynomial Owner) (width fuel : Nat) :
    evalPolynomial env (constructiveMergeSortAux le factors width fuel) =
      evalPolynomial env factors := by
  induction fuel generalizing factors width with
  | zero => rfl
  | succ fuel ih =>
      by_cases widthLt : width < factors.length
      · simp only [constructiveMergeSortAux, widthLt, ↓reduceIte]
        rw [ih, constructiveMergePass_evalPolynomial]
      · simp only [constructiveMergeSortAux, widthLt, ↓reduceIte]

private theorem constructiveMergeSort_evalPolynomial (env : Env Owner)
    (factors : Polynomial Owner) :
    evalPolynomial env (constructiveMergeSort termLe factors) = evalPolynomial env factors := by
  exact constructiveMergeSort_evalPolynomialAux env termLe factors 1 factors.length

def canonicalKey (key : MonomialKey Owner) : MonomialKey Owner :=
  { centralFactors := constructiveMergeSort ownerLe key.centralFactors
    orderedFactors := key.orderedFactors }

private def canonicalTerm (term : ExactTerm Owner) : ExactTerm Owner :=
  { term with key := canonicalKey term.key }

private def combineSame (coefficient : Int) (key : MonomialKey Owner) :
    Polynomial Owner → Int × Polynomial Owner
  | [] => (coefficient, [])
  | term :: terms =>
      if term.key = key then
        combineSame (coefficient + term.coefficient) key terms
      else
        (coefficient, term :: terms)

private theorem combineSame_tail_length (coefficient : Int) (key : MonomialKey Owner)
    (terms : Polynomial Owner) : (combineSame coefficient key terms).2.length ≤ terms.length := by
  induction terms generalizing coefficient with
  | nil => exact Nat.le_refl _
  | cons term terms ih =>
      by_cases equal : term.key = key
      · simp only [combineSame, equal, ↓reduceIte]
        exact Nat.le_trans (ih (coefficient + term.coefficient)) (Nat.le_succ _)
      · simp only [combineSame, equal, ↓reduceIte]
        exact Nat.le_refl _

private def combineCanonicalAux : Nat → Polynomial Owner → Polynomial Owner
  | 0, terms => terms
  | _fuel + 1, [] => []
  | fuel + 1, term :: terms =>
      let combined := combineSame term.coefficient term.key terms
      if combined.1 = 0 then
        combineCanonicalAux fuel combined.2
      else
        { coefficient := combined.1, key := term.key } :: combineCanonicalAux fuel combined.2

private def combineCanonical (terms : Polynomial Owner) : Polynomial Owner :=
  combineCanonicalAux terms.length terms

def canonicalPolynomial (terms : Polynomial Owner) : Polynomial Owner :=
  combineCanonical (constructiveMergeSort (fun left right => termLe left right)
    (terms.map canonicalTerm))

def CanonicalAgreement (left right : Polynomial Owner) : Prop :=
  canonicalPolynomial left = canonicalPolynomial right

instance canonicalAgreementDecidable (left right : Polynomial Owner) :
    Decidable (CanonicalAgreement left right) := by
  unfold CanonicalAgreement
  infer_instance

instance keyEquivalentDecidable (left right : MonomialKey Owner) :
    Decidable (KeyEquivalent left right) := by
  apply decidable_of_iff
    ((permCheck left.centralFactors right.centralFactors = true) ∧
      left.orderedFactors = right.orderedFactors)
  constructor
  · intro h
    exact ⟨permCheck_iff.mp h.1, h.2⟩
  · intro h
    exact ⟨permCheck_iff.mpr h.1, h.2⟩

theorem list_prod_perm {Factor : Type} (env : Env Factor) {left right : List Factor}
    (h : left.Perm right) : (left.map env).prod = (right.map env).prod := by
  induction h with
  | nil => rfl
  | cons a h ih => simp [ih]
  | swap a b l => simp [Int.mul_left_comm]
  | trans h₁ h₂ ih₁ ih₂ => exact ih₁.trans ih₂

theorem evalMonomial_of_key {Factor : Type} (env : Env Factor)
    {left right : MonomialKey Factor}
    (central : left.centralFactors.Perm right.centralFactors)
    (ordered : left.orderedFactors = right.orderedFactors) :
    evalMonomial env left = evalMonomial env right := by
  rw [evalMonomial, evalMonomial, ordered, list_prod_perm env central]

private theorem evalPolynomial_perm (env : Env Owner)
    {left right : Polynomial Owner} (permutation : left.Perm right) :
    evalPolynomial env left = evalPolynomial env right := by
  induction permutation with
  | nil => rfl
  | cons term permutation ih =>
      simp only [evalPolynomial]
      rw [ih]
  | swap left right terms =>
      simp only [evalPolynomial]
      simp [Int.add_left_comm]
  | trans first second ihFirst ihSecond => exact ihFirst.trans ihSecond

private theorem evalMonomial_canonicalKey (env : Env Owner) (key : MonomialKey Owner) :
    evalMonomial env (canonicalKey key) = evalMonomial env key := by
  unfold canonicalKey evalMonomial
  rw [constructiveMergeSort_prod env ownerLe key.centralFactors]

private theorem evalPolynomial_map_canonicalTerm (env : Env Owner)
    (terms : Polynomial Owner) :
    evalPolynomial env (terms.map canonicalTerm) = evalPolynomial env terms := by
  induction terms with
  | nil => rfl
  | cons term terms ih =>
      simp only [List.map, evalPolynomial]
      change term.coefficient * evalMonomial env (canonicalKey term.key) +
        evalPolynomial env (terms.map canonicalTerm) =
        term.coefficient * evalMonomial env term.key + evalPolynomial env terms
      rw [evalMonomial_canonicalKey env term.key, ih]

private theorem evalPolynomial_combineSame (env : Env Owner)
    (coefficient : Int) (key : MonomialKey Owner) (terms : Polynomial Owner) :
    coefficient * evalMonomial env key + evalPolynomial env terms =
      (combineSame coefficient key terms).1 * evalMonomial env key +
        evalPolynomial env (combineSame coefficient key terms).2 := by
  induction terms generalizing coefficient with
  | nil => rfl
  | cons term terms ih =>
      by_cases equal : term.key = key
      · simp only [combineSame, equal, ↓reduceIte, evalPolynomial]
        calc
          coefficient * evalMonomial env key +
              (term.coefficient * evalMonomial env key + evalPolynomial env terms) =
              (coefficient * evalMonomial env key +
                term.coefficient * evalMonomial env key) + evalPolynomial env terms := by
            simp [Int.add_assoc]
          _ = (coefficient + term.coefficient) * evalMonomial env key +
              evalPolynomial env terms := by rw [← Int.add_mul]
          _ = (combineSame (coefficient + term.coefficient) key terms).1 *
                evalMonomial env key +
              evalPolynomial env (combineSame (coefficient + term.coefficient) key terms).2 :=
            ih (coefficient + term.coefficient)
      · simp [combineSame, equal]

private theorem evalPolynomial_combineCanonicalAux (env : Env Owner)
    (fuel : Nat) (terms : Polynomial Owner) :
    evalPolynomial env (combineCanonicalAux fuel terms) = evalPolynomial env terms := by
  induction fuel generalizing terms with
  | zero => rfl
  | succ fuel ih =>
      cases terms with
      | nil => rfl
      | cons term terms =>
          let combined := combineSame term.coefficient term.key terms
          have combinedSound :
              term.coefficient * evalMonomial env term.key + evalPolynomial env terms =
                combined.1 * evalMonomial env term.key + evalPolynomial env combined.2 := by
            exact evalPolynomial_combineSame env term.coefficient term.key terms
          have recursive := ih combined.2
          by_cases zero : combined.1 = 0
          · simp only [combineCanonicalAux, combined, zero, ↓reduceIte]
            calc
              evalPolynomial env (combineCanonicalAux fuel combined.2) =
                  evalPolynomial env combined.2 := recursive
              _ = term.coefficient * evalMonomial env term.key + evalPolynomial env terms := by
                simp [zero] at combinedSound
                exact combinedSound.symm
          · simp only [combineCanonicalAux, combined, zero, ↓reduceIte, evalPolynomial]
            calc
              combined.1 * evalMonomial env term.key +
                  evalPolynomial env (combineCanonicalAux fuel combined.2) =
                  combined.1 * evalMonomial env term.key + evalPolynomial env combined.2 := by
                rw [recursive]
              _ = term.coefficient * evalMonomial env term.key + evalPolynomial env terms :=
                combinedSound.symm

private theorem evalPolynomial_combineCanonical (env : Env Owner)
    (terms : Polynomial Owner) :
    evalPolynomial env (combineCanonical terms) = evalPolynomial env terms := by
  exact evalPolynomial_combineCanonicalAux env terms.length terms

theorem canonicalPolynomial_eval (env : Env Owner) (terms : Polynomial Owner) :
    evalPolynomial env (canonicalPolynomial terms) = evalPolynomial env terms := by
  unfold canonicalPolynomial
  calc
    evalPolynomial env (combineCanonical (constructiveMergeSort termLe
        (terms.map canonicalTerm))) =
        evalPolynomial env (constructiveMergeSort termLe (terms.map canonicalTerm)) :=
      evalPolynomial_combineCanonical env _
    _ = evalPolynomial env (terms.map canonicalTerm) :=
      constructiveMergeSort_evalPolynomial env (terms.map canonicalTerm)
    _ = evalPolynomial env terms := evalPolynomial_map_canonicalTerm env terms

theorem canonicalAgreement_eval (env : Env Owner) (left right : Polynomial Owner)
    (agreement : CanonicalAgreement left right) :
    evalPolynomial env left = evalPolynomial env right := by
  rw [← canonicalPolynomial_eval env left, ← canonicalPolynomial_eval env right, agreement]

theorem evalMonomial_of_productKey (env : Env Owner)
    {left right output : MonomialKey Owner}
    (key : ProductKey left right output) :
    evalMonomial env output = evalMonomial env left * evalMonomial env right := by
  rw [evalMonomial, evalMonomial, evalMonomial]
  rw [list_prod_perm env key.1, key.2]
  simp [List.map_append, List.prod_append, Int.mul_assoc, Int.mul_left_comm]

theorem evalMonomial_of_contextKey (env : Env Owner)
    {source replacement output : MonomialKey Owner}
    {exteriorCentral : List Owner}
    {orderedStart orderedEndExclusive : Nat}
    (key : ContextKey
      { centralFactors := exteriorCentral, orderedFactors := source.orderedFactors }
      replacement output orderedStart orderedEndExclusive) :
    evalMonomial env output =
      evalMonomial env
        ((relationContext source exteriorCentral orderedStart orderedEndExclusive).plug
          replacement) := by
  apply evalMonomial_of_key env
  · change output.centralFactors.Perm (exteriorCentral ++ replacement.centralFactors)
    exact key.1
  · change output.orderedFactors =
      source.orderedFactors.take orderedStart ++ replacement.orderedFactors ++
        source.orderedFactors.drop orderedEndExclusive
    exact key.2

private def partitionBy (representative : MonomialKey Owner) :
    Polynomial Owner → Polynomial Owner × Polynomial Owner
  | [] => ([], [])
  | term :: terms =>
      if KeyEquivalent term.key representative then
        let partitioned := partitionBy representative terms
        (term :: partitioned.1, partitioned.2)
      else
        let partitioned := partitionBy representative terms
        (partitioned.1, term :: partitioned.2)

def aggregateCoeff (representative : MonomialKey Owner) (terms : Polynomial Owner) : Int :=
  ((partitionBy representative terms).1.map ExactTerm.coefficient).sum

/-! Coefficient support is derived from `left ++ right`; callers provide no representative,
    support list, or completeness artifact. -/
def CoefficientAgreement (left right : Polynomial Owner) : Prop :=
  ∀ term, term ∈ left ++ right →
    aggregateCoeff term.key left = aggregateCoeff term.key right

private theorem keyEquivalent_symm {left right : MonomialKey Owner}
    (h : KeyEquivalent left right) : KeyEquivalent right left := by
  exact ⟨h.1.symm, h.2.symm⟩

private theorem keyEquivalent_trans {left middle right : MonomialKey Owner}
    (h₁ : KeyEquivalent left middle) (h₂ : KeyEquivalent middle right) :
    KeyEquivalent left right := by
  exact ⟨h₁.1.trans h₂.1, h₁.2.trans h₂.2⟩

private theorem partitionBy_fst_mem (representative : MonomialKey Owner)
    {term : ExactTerm Owner} {terms : Polynomial Owner}
    (h : term ∈ (partitionBy representative terms).1) : term ∈ terms := by
  induction terms with
  | nil => simp [partitionBy] at h
  | cons head terms ih =>
      by_cases equivalent : KeyEquivalent head.key representative
      · simp only [partitionBy, equivalent, ↓reduceIte] at h
        simp only [List.mem_cons] at h
        rcases h with rfl | h
        · exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ (ih h)
      · simp only [partitionBy, equivalent, ↓reduceIte] at h
        change term ∈ (partitionBy representative terms).1 at h
        exact List.mem_cons_of_mem _ (ih h)

private theorem partitionBy_snd_mem (representative : MonomialKey Owner)
    {term : ExactTerm Owner} {terms : Polynomial Owner}
    (h : term ∈ (partitionBy representative terms).2) : term ∈ terms := by
  induction terms with
  | nil => simp [partitionBy] at h
  | cons head terms ih =>
      by_cases equivalent : KeyEquivalent head.key representative
      · simp only [partitionBy, equivalent, ↓reduceIte] at h
        change term ∈ (partitionBy representative terms).2 at h
        exact List.mem_cons_of_mem _ (ih h)
      · simp only [partitionBy, equivalent, ↓reduceIte] at h
        simp only [List.mem_cons] at h
        rcases h with rfl | h
        · exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ (ih h)

private theorem partitionBy_snd_not_equivalent (representative : MonomialKey Owner)
    {term : ExactTerm Owner} {terms : Polynomial Owner}
    (h : term ∈ (partitionBy representative terms).2) :
    ¬ KeyEquivalent term.key representative := by
  induction terms with
  | nil => simp [partitionBy] at h
  | cons head terms ih =>
      by_cases equivalent : KeyEquivalent head.key representative
      · simp only [partitionBy, equivalent, ↓reduceIte] at h
        change term ∈ (partitionBy representative terms).2 at h
        exact ih h
      · simp only [partitionBy, equivalent, ↓reduceIte] at h
        simp only [List.mem_cons] at h
        rcases h with h | h
        · simpa [h] using equivalent
        · exact ih h

private theorem partitionBy_snd_partitionBy_fst
    (representative key : MonomialKey Owner) (terms : Polynomial Owner)
    (notEquivalent : ¬ KeyEquivalent key representative) :
    (partitionBy key (partitionBy representative terms).2).1 =
      (partitionBy key terms).1 := by
  induction terms with
  | nil => simp [partitionBy]
  | cons head terms ih =>
      by_cases equivalent : KeyEquivalent head.key representative
      · have headNotEquivalent : ¬ KeyEquivalent key head.key := by
          intro keyHead
          exact notEquivalent (keyEquivalent_trans keyHead equivalent)
        have headKeyNotEquivalent : ¬ KeyEquivalent head.key key := by
          intro keyHead
          exact headNotEquivalent (keyEquivalent_symm keyHead)
        simp [partitionBy, equivalent, headKeyNotEquivalent, ih]
      · by_cases keyHead : KeyEquivalent head.key key
        · simp [partitionBy, equivalent, keyHead, ih]
        · simp [partitionBy, equivalent, keyHead, ih]

private theorem partitionBy_snd_partitionBy_fst_of_equivalent
    (representative key : MonomialKey Owner) (terms : Polynomial Owner)
    (equivalent : KeyEquivalent key representative) :
    (partitionBy key (partitionBy representative terms).2).1 = [] := by
  induction terms with
  | nil => simp [partitionBy]
  | cons head terms ih =>
      by_cases headEquivalent : KeyEquivalent head.key representative
      · simp [partitionBy, headEquivalent, ih]
      · have headKeyNotEquivalent : ¬ KeyEquivalent head.key key := by
          intro headKey
          exact headEquivalent (keyEquivalent_trans headKey equivalent)
        simp [partitionBy, headEquivalent, headKeyNotEquivalent, ih]

private theorem aggregateCoeff_snd_of_not_equivalent
    (representative key : MonomialKey Owner) (terms : Polynomial Owner)
    (notEquivalent : ¬ KeyEquivalent key representative) :
    aggregateCoeff key (partitionBy representative terms).2 =
      aggregateCoeff key terms := by
  rw [aggregateCoeff, partitionBy_snd_partitionBy_fst representative key terms notEquivalent,
    aggregateCoeff]

private theorem aggregateCoeff_snd_of_equivalent
    (representative key : MonomialKey Owner) (terms : Polynomial Owner)
    (equivalent : KeyEquivalent key representative) :
    aggregateCoeff key (partitionBy representative terms).2 = 0 := by
  rw [aggregateCoeff, partitionBy_snd_partitionBy_fst_of_equivalent representative key terms
    equivalent]
  rfl

private theorem coefficientAgreement_snd (representative : MonomialKey Owner)
    {left right : Polynomial Owner} (agreement : CoefficientAgreement left right) :
    CoefficientAgreement (partitionBy representative left).2
      (partitionBy representative right).2 := by
  intro term h
  rcases List.mem_append.mp h with hleft | hright
  · have sourceMembership : term ∈ left ++ right :=
      List.mem_append.mpr (Or.inl (partitionBy_snd_mem representative hleft))
    have rawAgreement := agreement term sourceMembership
    have notEquivalent := partitionBy_snd_not_equivalent representative hleft
    rw [aggregateCoeff_snd_of_not_equivalent representative term.key left notEquivalent,
      aggregateCoeff_snd_of_not_equivalent representative term.key right notEquivalent]
    exact rawAgreement

  · have sourceMembership : term ∈ left ++ right :=
      List.mem_append.mpr (Or.inr (partitionBy_snd_mem representative hright))
    have rawAgreement := agreement term sourceMembership
    have notEquivalent := partitionBy_snd_not_equivalent representative hright
    rw [aggregateCoeff_snd_of_not_equivalent representative term.key left notEquivalent,
      aggregateCoeff_snd_of_not_equivalent representative term.key right notEquivalent]
    exact rawAgreement

private theorem evalPolynomial_partition (env : Env Owner)
    (representative : MonomialKey Owner) (terms : Polynomial Owner) :
    evalPolynomial env terms =
      evalPolynomial env (partitionBy representative terms).1 +
        evalPolynomial env (partitionBy representative terms).2 := by
  induction terms with
  | nil => rfl
  | cons term terms ih =>
      simp only [partitionBy]
      split
      · simp only [evalPolynomial]
        rw [ih]
        simp [Int.add_assoc]
      · simp only [evalPolynomial]
        rw [ih]
        simp [Int.add_left_comm]

private theorem evalPolynomial_partition_matches (env : Env Owner)
    (representative : MonomialKey Owner) (terms : Polynomial Owner) :
    evalPolynomial env (partitionBy representative terms).1 =
      aggregateCoeff representative terms * evalMonomial env representative := by
  induction terms with
  | nil => simp [aggregateCoeff, partitionBy, evalPolynomial]
  | cons term terms ih =>
      by_cases h : KeyEquivalent term.key representative
      · simp [partitionBy, h, aggregateCoeff, evalPolynomial, ih,
          evalMonomial_of_key env h.1 h.2, Int.add_mul]
      · simp [partitionBy, h, aggregateCoeff, ih]

private theorem partitionBy_snd_length_le (representative : MonomialKey Owner)
    (terms : Polynomial Owner) : (partitionBy representative terms).2.length ≤ terms.length := by
  induction terms with
  | nil => exact Nat.zero_le _
  | cons term terms ih =>
      by_cases equivalent : KeyEquivalent term.key representative
      · simp only [partitionBy, if_pos equivalent]
        change (partitionBy representative terms).2.length ≤ terms.length + 1
        exact Nat.le_trans ih (Nat.le_succ _)
      · simp only [partitionBy, if_neg equivalent]
        change (term :: (partitionBy representative terms).2).length ≤ terms.length + 1
        exact Nat.succ_le_succ ih

private theorem partitionBy_snd_length_lt_of_equivalent
    (representative : MonomialKey Owner) (term : ExactTerm Owner)
    (terms : Polynomial Owner) (equivalent : KeyEquivalent term.key representative) :
    (partitionBy representative (term :: terms)).2.length < (term :: terms).length := by
  simp only [partitionBy, equivalent, ↓reduceIte]
  dsimp
  have bound := partitionBy_snd_length_le representative terms
  exact Nat.lt_succ_of_le bound

private theorem keyEquivalent_refl (key : MonomialKey Owner) : KeyEquivalent key key :=
  ⟨List.Perm.refl _, rfl⟩

private theorem coefficientAgreement_partitionBy_snd
    (representative : MonomialKey Owner) (left right : Polynomial Owner)
    (agreement : CoefficientAgreement left right) :
    CoefficientAgreement (partitionBy representative left).2
      (partitionBy representative right).2 := by
  intro term termMem
  rcases List.mem_append.mp termMem with termMem | termMem
  · have original := partitionBy_snd_mem representative termMem
    have notEquivalent := partitionBy_snd_not_equivalent representative termMem
    rw [aggregateCoeff_snd_of_not_equivalent representative term.key left notEquivalent,
      aggregateCoeff_snd_of_not_equivalent representative term.key right notEquivalent]
    exact agreement term (List.mem_append_left _ original)
  · have original := partitionBy_snd_mem representative termMem
    have notEquivalent := partitionBy_snd_not_equivalent representative termMem
    rw [aggregateCoeff_snd_of_not_equivalent representative term.key left notEquivalent,
      aggregateCoeff_snd_of_not_equivalent representative term.key right notEquivalent]
    exact agreement term (List.mem_append_right _ original)

theorem coefficientAgreement_eval (env : Env Owner) (left right : Polynomial Owner)
    (agreement : CoefficientAgreement left right) :
    evalPolynomial env left = evalPolynomial env right := by
  let motive : Nat → Prop := fun total =>
    ∀ (left right : Polynomial Owner), left.length + right.length = total →
      CoefficientAgreement left right → evalPolynomial env left = evalPolynomial env right
  refine Nat.strongRecOn (motive := motive) (left.length + right.length) ?_ left right rfl agreement
  intro total inductionHypothesis left right totalEq agreement
  cases left with
  | nil =>
      cases right with
      | nil => rfl
      | cons term terms =>
          let representative := term.key
          have representativeAgreement := agreement term (by simp)
          have aggregateRightZero : aggregateCoeff representative (term :: terms) = 0 := by
            simpa [aggregateCoeff, partitionBy] using representativeAgreement.symm
          have residualAgreement := coefficientAgreement_partitionBy_snd representative
            [] (term :: terms) agreement
          have residualLength :
              (partitionBy representative []).2.length +
                (partitionBy representative (term :: terms)).2.length < total := by
            have bound := partitionBy_snd_length_lt_of_equivalent representative term terms
              (keyEquivalent_refl term.key)
            have emptyLength : (partitionBy representative []).2.length = 0 := by
              simp [partitionBy]
            rw [emptyLength]
            simp only [representative] at bound
            have termEq : (term :: terms).length = total := by simpa using totalEq
            have termLe : (term :: terms).length ≤ total := Nat.le_of_eq termEq
            exact Nat.lt_of_lt_of_le (by simpa using bound) termLe
          have recursive := inductionHypothesis _ residualLength _ _ rfl residualAgreement
          calc
            evalPolynomial env [] =
                evalPolynomial env (partitionBy representative []).1 +
                  evalPolynomial env (partitionBy representative []).2 :=
              evalPolynomial_partition env representative []
            _ = evalPolynomial env (partitionBy representative (term :: terms)).1 +
                  evalPolynomial env (partitionBy representative []).2 := by
              rw [show evalPolynomial env (partitionBy representative (term :: terms)).1 =
                aggregateCoeff representative (term :: terms) *
                  evalMonomial env representative by
                exact evalPolynomial_partition_matches env representative (term :: terms)]
              rw [aggregateRightZero]
              simp [partitionBy, evalPolynomial]
            _ = evalPolynomial env (partitionBy representative (term :: terms)).1 +
                  evalPolynomial env (partitionBy representative (term :: terms)).2 := by
              rw [recursive]
            _ = evalPolynomial env (term :: terms) := by
              symm
              exact evalPolynomial_partition env representative (term :: terms)
  | cons term terms =>
      let representative := term.key
      have representativeAgreement := agreement term (by simp)
      have residualAgreement := coefficientAgreement_partitionBy_snd representative
        (term :: terms) right agreement
      have residualLength :
          (partitionBy representative (term :: terms)).2.length +
            (partitionBy representative right).2.length < total := by
        have leftBound := partitionBy_snd_length_lt_of_equivalent representative term terms
          (keyEquivalent_refl term.key)
        have rightBound := partitionBy_snd_length_le representative right
        simp only [representative] at leftBound rightBound ⊢
        exact Nat.lt_of_lt_of_eq
          (Nat.add_lt_add_of_lt_of_le leftBound rightBound) totalEq
      have recursive := inductionHypothesis _ residualLength _ _ rfl residualAgreement
      calc
        evalPolynomial env (term :: terms) =
            evalPolynomial env (partitionBy representative (term :: terms)).1 +
              evalPolynomial env (partitionBy representative (term :: terms)).2 :=
          evalPolynomial_partition env representative _
        _ = aggregateCoeff representative (term :: terms) * evalMonomial env representative +
              evalPolynomial env (partitionBy representative (term :: terms)).2 := by
          rw [evalPolynomial_partition_matches]
        _ = aggregateCoeff representative right * evalMonomial env representative +
              evalPolynomial env (partitionBy representative (term :: terms)).2 := by
          rw [representativeAgreement]
        _ = evalPolynomial env (partitionBy representative right).1 +
              evalPolynomial env (partitionBy representative (term :: terms)).2 := by
          rw [evalPolynomial_partition_matches]
        _ = evalPolynomial env (partitionBy representative right).1 +
              evalPolynomial env (partitionBy representative right).2 := by
          rw [recursive]
        _ = evalPolynomial env right := by
          symm
          exact evalPolynomial_partition env representative right

private theorem evalPolynomial_append (env : Env Owner)
    (left right : Polynomial Owner) :
    evalPolynomial env (left ++ right) = evalPolynomial env left + evalPolynomial env right := by
  induction left with
  | nil => simp [evalPolynomial]
  | cons term left ih =>
      simp only [List.cons_append, evalPolynomial]
      rw [ih]
      simp [Int.add_assoc]

private theorem evalPolynomial_negate (env : Env Owner) (terms : Polynomial Owner) :
    evalPolynomial env (negate terms) = -evalPolynomial env terms := by
  induction terms with
  | nil => rfl
  | cons term terms ih =>
      simp only [negate, evalPolynomial]
      rw [ih]
      simp [Int.neg_mul, Int.neg_add]

theorem addResultSound (env : Env Owner) (left right output : Polynomial Owner)
    (agreement : CoefficientAgreement output (add left right)) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  rw [coefficientAgreement_eval env output (add left right) agreement,
    show add left right = left ++ right by rfl,
    evalPolynomial_append env left right]

theorem subResultSound (env : Env Owner) (left right output : Polynomial Owner)
    (agreement : CoefficientAgreement output (subtract left right)) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  rw [coefficientAgreement_eval env output (subtract left right) agreement,
    show subtract left right = left ++ negate right by rfl,
    evalPolynomial_append, evalPolynomial_negate]
  rfl

def productPoly (left right : Polynomial Owner) (leftScalar rightScalar : Bool) :
    Polynomial Owner :=
  left.flatMap (fun leftTerm =>
    right.map (fun rightTerm =>
      operatorProductContribution leftTerm rightTerm leftScalar rightScalar))

private theorem evalMonomial_of_scalarAction (env : Env Owner) (key : MonomialKey Owner) :
    evalMonomial env (scalarActionKey key) = evalMonomial env key := by
  rw [evalMonomial, evalMonomial]
  rw [scalarActionKey_central, scalarActionKey_ordered]
  rw [show CentralNormalizer.normalize (key.centralFactors ++ key.orderedFactors) =
    key.centralFactors ++ key.orderedFactors by rfl]
  simp [List.map_append, List.prod_append]

private theorem productKey_product (left right : MonomialKey Owner) :
    ProductKey left right (MonomialKey.product left right) := by
  constructor
  · rw [product_central]
    change (left.centralFactors ++ right.centralFactors).Perm _
    exact List.Perm.refl _
  · exact product_ordered left right

private theorem evalMonomial_of_scalarProductKey (env : Env Owner)
    (left right : MonomialKey Owner) (leftScalar rightScalar : Bool) :
    evalMonomial env (scalarProductKey left right leftScalar rightScalar) =
      evalMonomial env left * evalMonomial env right := by
  cases leftScalar <;> cases rightScalar <;> simp [scalarProductKey]
  all_goals
    rw [evalMonomial_of_productKey env (productKey_product _ _)] <;>
      simp [evalMonomial_of_scalarAction]

private theorem evalOperatorProductContribution (env : Env Owner)
    (left right : ExactTerm Owner) (leftScalar rightScalar : Bool) :
    (operatorProductContribution left right leftScalar rightScalar).coefficient *
        evalMonomial env (operatorProductContribution left right leftScalar rightScalar).key =
      (left.coefficient * evalMonomial env left.key) *
        (right.coefficient * evalMonomial env right.key) := by
  rw [operatorProductContribution_coefficient, operatorProductContribution_key,
    evalMonomial_of_scalarProductKey]
  simp [Int.mul_assoc, Int.mul_left_comm]

private theorem evalPolynomial_productMap (env : Env Owner)
    (leftTerm : ExactTerm Owner) (right : Polynomial Owner)
    (leftScalar rightScalar : Bool) :
    evalPolynomial env
        (right.map (fun rightTerm =>
          operatorProductContribution leftTerm rightTerm leftScalar rightScalar)) =
      (leftTerm.coefficient * evalMonomial env leftTerm.key) *
        evalPolynomial env right := by
  induction right with
  | nil => simp [evalPolynomial]
  | cons rightTerm rightTail ih =>
      simp only [List.map, evalPolynomial]
      rw [ih]
      rw [evalOperatorProductContribution]
      simp [Int.mul_assoc, Int.mul_left_comm, Int.mul_add]

private theorem evalPolynomial_productPoly (env : Env Owner)
    (left right : Polynomial Owner) (leftScalar rightScalar : Bool) :
    evalPolynomial env (productPoly left right leftScalar rightScalar) =
      evalPolynomial env left * evalPolynomial env right := by
  induction left with
  | nil => simp [productPoly, evalPolynomial]
  | cons leftTerm leftTail ih =>
      change evalPolynomial env
          (right.map (fun rightTerm =>
            operatorProductContribution leftTerm rightTerm leftScalar rightScalar) ++
            productPoly leftTail right leftScalar rightScalar) = _
      rw [evalPolynomial_append, evalPolynomial_productMap, ih]
      simp only [evalPolynomial]
      simp [Int.mul_assoc, Int.mul_comm, Int.mul_add]

theorem productResultSound (env : Env Owner) (left right output : Polynomial Owner)
    (leftScalar rightScalar : Bool)
    (agreement : CoefficientAgreement output (productPoly left right leftScalar rightScalar)) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  rw [coefficientAgreement_eval env output
      (productPoly left right leftScalar rightScalar) agreement,
    evalPolynomial_productPoly]

theorem addCanonicalResultSound (env : Env Owner) (left right output : Polynomial Owner)
    (agreement : CanonicalAgreement output (add left right)) :
    evalPolynomial env output = evalPolynomial env left + evalPolynomial env right := by
  rw [canonicalAgreement_eval env output (add left right) agreement,
    show add left right = left ++ right by rfl,
    evalPolynomial_append env left right]

theorem subCanonicalResultSound (env : Env Owner) (left right output : Polynomial Owner)
    (agreement : CanonicalAgreement output (subtract left right)) :
    evalPolynomial env output = evalPolynomial env left - evalPolynomial env right := by
  rw [canonicalAgreement_eval env output (subtract left right) agreement,
    show subtract left right = left ++ negate right by rfl,
    evalPolynomial_append, evalPolynomial_negate]
  rfl

theorem productCanonicalResultSound (env : Env Owner) (left right output : Polynomial Owner)
    (leftScalar rightScalar : Bool)
    (agreement : CanonicalAgreement output
      (productPoly left right leftScalar rightScalar)) :
    evalPolynomial env output = evalPolynomial env left * evalPolynomial env right := by
  rw [canonicalAgreement_eval env output
      (productPoly left right leftScalar rightScalar) agreement,
    evalPolynomial_productPoly]

def relationPoly (accumulator : Polynomial Owner) (sourceKey : MonomialKey Owner)
    (context : MonomialContext Owner) (outerCoefficient : Int)
    (rhs : Polynomial Owner) : Polynomial Owner :=
  add (subtract accumulator
      [{ coefficient := outerCoefficient, key := sourceKey }])
    (relationReplacement context outerCoefficient rhs)

def contextMultiplier (env : Env Owner) (context : MonomialContext Owner) : Int :=
  (context.exteriorCentral.map env).prod *
    (context.prefixFactors.map env).prod * (context.suffixFactors.map env).prod

private theorem evalMonomial_contextPlug (env : Env Owner)
    (context : MonomialContext Owner) (key : MonomialKey Owner) :
    evalMonomial env (context.plug key) =
      contextMultiplier env context * evalMonomial env key := by
  rw [evalMonomial, contextMultiplier, context_plug_central, context_plug_ordered]
  rw [show CentralNormalizer.normalize (context.exteriorCentral ++ key.centralFactors) =
    context.exteriorCentral ++ key.centralFactors by rfl]
  simp [evalMonomial, List.map_append, List.prod_append, Int.mul_assoc,
    Int.mul_left_comm, Int.mul_comm]

private theorem evalPolynomial_eventReplay (env : Env Owner) (terms : Polynomial Owner) :
    EventReplay.evaluatePolynomial (fun key => evalMonomial env key) terms =
      evalPolynomial env terms := by
  induction terms with
  | nil => rfl
  | cons term terms ih => simp [EventReplay.evaluatePolynomial, evalPolynomial, ih]

theorem relationResultSound (modulus : Nat) (env : Env Owner)
    (accumulator : Polynomial Owner) (sourceKey lhsKey : MonomialKey Owner)
    (exteriorCentral : List Owner) (orderedStart orderedEndExclusive : Nat)
    (outerCoefficient : Int) (rhs output : Polynomial Owner)
    (sourceKeyEquivalent :
      KeyEquivalent sourceKey
        ((relationContext sourceKey exteriorCentral orderedStart orderedEndExclusive).plug lhsKey))
    (baseRelation :
      evalMonomial env lhsKey % Int.ofNat modulus =
        evalPolynomial env rhs % Int.ofNat modulus)
    (agreement : CoefficientAgreement output
      (relationPoly accumulator sourceKey
        (relationContext sourceKey exteriorCentral orderedStart orderedEndExclusive)
        outerCoefficient rhs)) :
    evalPolynomial env output % Int.ofNat modulus =
      evalPolynomial env accumulator % Int.ofNat modulus := by
  let context := relationContext sourceKey exteriorCentral orderedStart orderedEndExclusive
  have contextSound : ∀ key,
      (fun key => evalMonomial env key) (context.plug key) =
        contextMultiplier env context * (fun key => evalMonomial env key) key := by
    intro key
    exact evalMonomial_contextPlug env context key
  let lhsPolynomial : Polynomial Owner :=
    [{ coefficient := 1, key := lhsKey }]
  have baseRelation' :
      EventReplay.evaluatePolynomial (fun key => evalMonomial env key) lhsPolynomial %
          Int.ofNat modulus =
        EventReplay.evaluatePolynomial (fun key => evalMonomial env key) rhs %
          Int.ofNat modulus := by
    calc
      EventReplay.evaluatePolynomial (fun key => evalMonomial env key) lhsPolynomial %
            Int.ofNat modulus = evalMonomial env lhsKey % Int.ofNat modulus := by
        simp [lhsPolynomial, EventReplay.evaluatePolynomial]
      _ = evalPolynomial env rhs % Int.ofNat modulus := baseRelation
      _ = EventReplay.evaluatePolynomial (fun key => evalMonomial env key) rhs %
            Int.ofNat modulus := by
        rw [evalPolynomial_eventReplay]
  have replacementCongruence := EventReplay.relationReplacement_modular modulus
    (fun key => evalMonomial env key) context (contextMultiplier env context) outerCoefficient
    lhsPolynomial rhs contextSound baseRelation'
  rw [evalPolynomial_eventReplay, evalPolynomial_eventReplay] at replacementCongruence
  have lhsReplacement :
      evalPolynomial env (relationReplacement context outerCoefficient lhsPolynomial) =
        outerCoefficient * evalMonomial env sourceKey := by
    simp [lhsPolynomial, relationReplacement, scalePolynomial, contextualize, evalPolynomial]
    rw [evalMonomial_of_key env sourceKeyEquivalent.1 sourceKeyEquivalent.2]
  have outputEval := coefficientAgreement_eval env output
    (relationPoly accumulator sourceKey context outerCoefficient rhs) agreement
  have relationEvalModular :
      evalPolynomial env (relationPoly accumulator sourceKey context outerCoefficient rhs) %
          Int.ofNat modulus = evalPolynomial env accumulator % Int.ofNat modulus := by
    rw [relationPoly]
    simp only [add, subtract]
    rw [evalPolynomial_append env
        (accumulator ++ negate [{ coefficient := outerCoefficient, key := sourceKey }])
        (relationReplacement context outerCoefficient rhs),
      evalPolynomial_append env accumulator
        (negate [{ coefficient := outerCoefficient, key := sourceKey }]),
      evalPolynomial_negate]
    simp only [evalPolynomial]
    simp only [Int.add_zero]
    have replacementDifference :
        (evalPolynomial env (relationReplacement context outerCoefficient rhs) -
            outerCoefficient * evalMonomial env sourceKey) %
          Int.ofNat modulus = 0 := by
      have difference :=
        Int.emod_eq_emod_iff_emod_sub_eq_zero.mp replacementCongruence.symm
      rw [lhsReplacement] at difference
      exact difference
    calc
      (evalPolynomial env accumulator +
          -(outerCoefficient * evalMonomial env sourceKey) +
          evalPolynomial env (relationReplacement context outerCoefficient rhs)) %
            Int.ofNat modulus =
          (evalPolynomial env accumulator +
            (evalPolynomial env (relationReplacement context outerCoefficient rhs) -
              outerCoefficient * evalMonomial env sourceKey)) %
            Int.ofNat modulus := by
        congr 1
        simp [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm]
      _ = (evalPolynomial env accumulator % Int.ofNat modulus +
          (evalPolynomial env (relationReplacement context outerCoefficient rhs) -
            outerCoefficient * evalMonomial env sourceKey) % Int.ofNat modulus) %
            Int.ofNat modulus := by
        rw [Int.add_emod]
      _ = evalPolynomial env accumulator % Int.ofNat modulus := by
        rw [replacementDifference]
        simp
  rw [outputEval, relationEvalModular]

theorem relationCanonicalResultSound (modulus : Nat) (env : Env Owner)
    (accumulator : Polynomial Owner) (sourceKey lhsKey : MonomialKey Owner)
    (exteriorCentral : List Owner) (orderedStart orderedEndExclusive : Nat)
    (outerCoefficient : Int) (rhs output : Polynomial Owner)
    (sourceKeyEquivalent :
      KeyEquivalent sourceKey
        ((relationContext sourceKey exteriorCentral orderedStart orderedEndExclusive).plug lhsKey))
    (baseRelation :
      evalMonomial env lhsKey % Int.ofNat modulus =
        evalPolynomial env rhs % Int.ofNat modulus)
    (agreement : CanonicalAgreement output
      (relationPoly accumulator sourceKey
        (relationContext sourceKey exteriorCentral orderedStart orderedEndExclusive)
        outerCoefficient rhs)) :
    evalPolynomial env output % Int.ofNat modulus =
      evalPolynomial env accumulator % Int.ofNat modulus := by
  let context := relationContext sourceKey exteriorCentral orderedStart orderedEndExclusive
  have contextSound : ∀ key,
      (fun key => evalMonomial env key) (context.plug key) =
        contextMultiplier env context * (fun key => evalMonomial env key) key := by
    intro key
    exact evalMonomial_contextPlug env context key
  let lhsPolynomial : Polynomial Owner :=
    [{ coefficient := 1, key := lhsKey }]
  have baseRelation' :
      EventReplay.evaluatePolynomial (fun key => evalMonomial env key) lhsPolynomial %
          Int.ofNat modulus =
        EventReplay.evaluatePolynomial (fun key => evalMonomial env key) rhs %
          Int.ofNat modulus := by
    calc
      EventReplay.evaluatePolynomial (fun key => evalMonomial env key) lhsPolynomial %
            Int.ofNat modulus = evalMonomial env lhsKey % Int.ofNat modulus := by
        simp [lhsPolynomial, EventReplay.evaluatePolynomial]
      _ = evalPolynomial env rhs % Int.ofNat modulus := baseRelation
      _ = EventReplay.evaluatePolynomial (fun key => evalMonomial env key) rhs %
            Int.ofNat modulus := by
        rw [evalPolynomial_eventReplay]
  have replacementCongruence := EventReplay.relationReplacement_modular modulus
    (fun key => evalMonomial env key) context (contextMultiplier env context) outerCoefficient
    lhsPolynomial rhs contextSound baseRelation'
  rw [evalPolynomial_eventReplay, evalPolynomial_eventReplay] at replacementCongruence
  have lhsReplacement :
      evalPolynomial env (relationReplacement context outerCoefficient lhsPolynomial) =
        outerCoefficient * evalMonomial env sourceKey := by
    simp [lhsPolynomial, relationReplacement, scalePolynomial, contextualize, evalPolynomial]
    rw [evalMonomial_of_key env sourceKeyEquivalent.1 sourceKeyEquivalent.2]
  have outputEval := canonicalAgreement_eval env output
    (relationPoly accumulator sourceKey context outerCoefficient rhs) agreement
  have relationEvalModular :
      evalPolynomial env (relationPoly accumulator sourceKey context outerCoefficient rhs) %
          Int.ofNat modulus = evalPolynomial env accumulator % Int.ofNat modulus := by
    rw [relationPoly]
    simp only [add, subtract]
    rw [evalPolynomial_append env
        (accumulator ++ negate [{ coefficient := outerCoefficient, key := sourceKey }])
        (relationReplacement context outerCoefficient rhs),
      evalPolynomial_append env accumulator
        (negate [{ coefficient := outerCoefficient, key := sourceKey }]),
      evalPolynomial_negate]
    simp only [evalPolynomial]
    simp only [Int.add_zero]
    have replacementDifference :
        (evalPolynomial env (relationReplacement context outerCoefficient rhs) -
            outerCoefficient * evalMonomial env sourceKey) %
          Int.ofNat modulus = 0 := by
      have difference :=
        Int.emod_eq_emod_iff_emod_sub_eq_zero.mp replacementCongruence.symm
      rw [lhsReplacement] at difference
      exact difference
    calc
      (evalPolynomial env accumulator +
          -(outerCoefficient * evalMonomial env sourceKey) +
          evalPolynomial env (relationReplacement context outerCoefficient rhs)) %
            Int.ofNat modulus =
          (evalPolynomial env accumulator +
            (evalPolynomial env (relationReplacement context outerCoefficient rhs) -
              outerCoefficient * evalMonomial env sourceKey)) %
            Int.ofNat modulus := by
        congr 1
        simp [Int.sub_eq_add_neg, Int.add_assoc, Int.add_left_comm, Int.add_comm]
      _ = (evalPolynomial env accumulator % Int.ofNat modulus +
          (evalPolynomial env (relationReplacement context outerCoefficient rhs) -
            outerCoefficient * evalMonomial env sourceKey) % Int.ofNat modulus) %
            Int.ofNat modulus := by
        rw [Int.add_emod]
      _ = evalPolynomial env accumulator % Int.ofNat modulus := by
        rw [replacementDifference]
        simp
  rw [outputEval, relationEvalModular]

/- The coefficient-transfer primitives are intentionally kept in `EventReplay`.  Tall generated
   proofs use those soundness lemmas directly, so this layer does not add same-shaped aliases. -/

def preFoldBound (summaryActual summaryBound : Nat)
    (survivorContributions survivorBounds : List Nat) : Prop :=
  summaryActual + survivorContributions.sum ≤ summaryBound + survivorBounds.sum

theorem preFoldSound (rootTerms terms : Polynomial Owner)
    (termsExact : terms = rootTerms)
    {summaryActual summaryBound : Nat}
    (rootRemainderBound : summaryActual ≤ summaryBound)
    {survivorContributions survivorBounds : List Nat}
    (survivors : List.Forall₂ (fun actual bound => actual ≤ bound)
      survivorContributions survivorBounds) :
    terms = rootTerms ∧ preFoldBound summaryActual summaryBound
      survivorContributions survivorBounds := by
  have survivorSum : survivorContributions.sum ≤ survivorBounds.sum := by
    induction survivors with
    | nil => exact Nat.zero_le _
    | cons head tail ih => exact Nat.add_le_add head ih
  exact ⟨termsExact, Nat.add_le_add rootRemainderBound survivorSum⟩

def boundInterprets (modulus : Nat) (bound : Bound) (value : Int) : Prop :=
  match bound with
  | .exactZero => centeredNorm modulus value = 0
  | .finite maximum => centeredNorm modulus value ≤ maximum
  | .large => True
  | .missing => False

def coeffClassToTallBound : CoeffClass → Bound
  | .exactZero => .exactZero
  | .finite maximum => .finite maximum.val
  | .large => .large

/-- The recorded row bound may be tighter than the conservative bound replayed from its
    producer. This relation validates that modeling refinement; coefficient soundness always
    comes from the producer bound. -/
def RecordedBoundRefines : Bound → CoeffClass → Prop
  | .exactZero, _ => True
  | .finite recorded, .exactZero => recorded = 0
  | .finite recorded, .finite produced => recorded ≤ produced.val
  | .finite _, .large => True
  | .large, .large => True
  | _, _ => False

theorem coeffClassInterprets_to_boundInterprets {modulus : Nat} {value : Int}
    {bound : CoeffClass} (sound : bound.Interprets (centeredNorm modulus value)) :
    boundInterprets modulus (coeffClassToTallBound bound) value := by
  cases bound with
  | exactZero => exact sound
  | finite maximum => exact sound
  | large => trivial

def addKnownList : List CoeffClass → CoeffClass
  | [] => .exactZero
  | bound :: bounds => addKnown bound (addKnownList bounds)

theorem addKnownList_sound {bounds : List CoeffClass} {actuals : List Nat}
    (sound : List.Forall₂ (fun bound actual => bound.Interprets actual) bounds actuals) :
    (addKnownList bounds).Interprets actuals.sum := by
  induction sound with
  | nil => simp [addKnownList, CoeffClass.Interprets]
  | cons head tail ih =>
      simpa [addKnownList] using addKnown_sound head ih

/-- An event-level claim; coefficient results do not create entries in `Env`. -/
inductive ValueClaim (Factor : Type) where
  | exact (terms : Polynomial Factor) (summary : Bound)
  | coefficient (bound : Bound)

def ValueClaim.Interprets {Factor : Type} (modulus : Nat) (env : Env Factor)
    (actual : Int) : ValueClaim Factor → Prop
  | .exact terms summary =>
      ∃ remainder,
        (actual - evalPolynomial env terms) % Int.ofNat modulus = remainder % Int.ofNat modulus ∧
          boundInterprets modulus summary remainder
  | .coefficient bound => boundInterprets modulus bound actual

/- A reached proof supplies the remainder congruence; this constructor turns that ABI fact into
   the event-level claim without introducing an evaluator or an owner-wide value assumption. -/
theorem exactValueClaim_of_remainder {Factor : Type} (modulus : Nat) (env : Env Factor)
    (actual : Int) (terms : Polynomial Factor) (maximum : Nat) (remainder : Int)
    (congruence :
      (actual - evalPolynomial env terms) % Int.ofNat modulus = remainder % Int.ofNat modulus)
    (remainderBound : centeredNorm modulus remainder ≤ maximum) :
    ValueClaim.Interprets modulus env actual (.exact terms (.finite maximum)) := by
  exact ⟨remainder, congruence, remainderBound⟩

theorem exactValueClaim_of_coeffClass {Factor : Type} (modulus : Nat) (env : Env Factor)
    (actual : Int) (terms : Polynomial Factor) (bound : CoeffClass) (remainder : Int)
    (congruence :
      (actual - evalPolynomial env terms) % Int.ofNat modulus = remainder % Int.ofNat modulus)
    (remainderSound : bound.Interprets (centeredNorm modulus remainder)) :
    ValueClaim.Interprets modulus env actual (.exact terms (coeffClassToTallBound bound)) := by
  exact ⟨remainder, congruence, coeffClassInterprets_to_boundInterprets remainderSound⟩

/-- A reached authority transfer and the unique Result row that records its coefficient bound. -/
inductive AuthorityLeafAt (history : EventHistory) :
    Nat → Nat → Nat → Owner → Authority → CoeffClass → Prop where
  | resultCoefficient {producerEvent resultEvent frameStart : Nat} {owner : Owner}
      {authority : Authority} {bound : CoeffClass}
      (adjacent : producerEvent + 1 = resultEvent)
      (transferRow : history.lookup producerEvent = some
        ⟨.boundTransfer owner (.authority authority), frameStart⟩)
      (resultRow : history.lookup resultEvent = some
        ⟨.resultCoefficient owner (coeffClassToTallBound bound), frameStart⟩) :
      AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound
  | resultExact {producerEvent resultEvent frameStart : Nat} {owner : Owner}
      {authority : Authority} {bound : CoeffClass} {terms : List Term}
      {recordedCoefficientBound summary : Bound} {summaryProducer : Option Nat}
      (producerBefore : producerEvent < resultEvent)
      (transferRow : history.lookup producerEvent = some
        ⟨.boundTransfer owner (.authority authority), frameStart⟩)
      (resultRow : history.lookup resultEvent = some
        ⟨.resultExact owner terms recordedCoefficientBound producerEvent summary
          summaryProducer, frameStart⟩)
      (refines : RecordedBoundRefines recordedCoefficientBound bound) :
      AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound

/-- The only semantic assumption used by authority coefficient leaves, indexed by their full
    history evidence rather than by a freely supplied magnitude or soundness proposition. -/
structure AuthorityWitness (history : EventHistory) where
  authorityMagnitude : Nat → Nat
  authorityBound : ∀ producerEvent resultEvent frameStart owner authority bound,
    AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound →
      bound.Interprets (authorityMagnitude resultEvent)

/-! Bound derivations and result projections are mutually indexed by concrete history rows.
    Every coefficient projection contains the `BoundDerivedAt` selected by the recorded producer;
    no constructor accepts a standalone soundness proposition. -/
mutual
  inductive ProjectedBoundAt (history : EventHistory) :
      Nat → Owner → Option (List Term) → Projection → CoeffClass → Nat → Prop where
    | resultCoefficient {resultEvent producerEvent frameStart : Nat} {owner : Owner}
        {rule : BoundRule}
        {bound : CoeffClass} {actualMagnitude : Nat}
        (adjacent : producerEvent + 1 = resultEvent)
        (row : history.lookup resultEvent = some
          ⟨.resultCoefficient owner (coeffClassToTallBound bound), frameStart⟩)
        (derived : BoundDerivedAt history producerEvent frameStart owner rule bound actualMagnitude) :
        ProjectedBoundAt history resultEvent owner none .coefficient bound actualMagnitude
    | resultExactCoefficient {resultEvent : Nat} {owner : Owner} {terms : List Term}
        {coefficientProducer frameStart : Nat}
        {summary : Bound} {summaryProducer : Option Nat} {rule : BoundRule}
        {recordedCoefficientBound : Bound} {producerBound : CoeffClass}
        {actualMagnitude : Nat}
        (row : history.lookup resultEvent = some
          ⟨.resultExact owner terms recordedCoefficientBound coefficientProducer
            summary summaryProducer, frameStart⟩)
        (refines : RecordedBoundRefines recordedCoefficientBound producerBound)
        (derived : BoundDerivedAt history coefficientProducer frameStart owner rule
          producerBound actualMagnitude) :
        ProjectedBoundAt history resultEvent owner (some terms) .coefficient producerBound
          actualMagnitude
    | resultExactSummary {resultEvent producerEvent frameStart coefficientProducer : Nat}
        {owner : Owner} {terms : List Term} {recordedCoefficientBound : Bound}
        {rule : BoundRule} {producerBound : CoeffClass} {actualMagnitude : Nat}
        (row : history.lookup resultEvent = some
          ⟨.resultExact owner terms recordedCoefficientBound coefficientProducer
            (coeffClassToTallBound producerBound) (some producerEvent), frameStart⟩)
        (derived : BoundDerivedAt history producerEvent frameStart owner rule producerBound
          actualMagnitude) :
        ProjectedBoundAt history resultEvent owner (some terms) .summary producerBound
          actualMagnitude

  inductive BoundInputAt (history : EventHistory) :
      Owner → ValueRef → CoeffClass → Nat → Prop where
    | result {consumer : Owner} {resultEvent : Nat} {resultOwner : Owner}
        {rawTerms : Option (List Term)}
        {bound : CoeffClass} {actualMagnitude : Nat}
        (scope : resultOwner.scope = consumer.scope)
        (projected : ProjectedBoundAt history resultEvent resultOwner rawTerms .coefficient
          bound actualMagnitude) :
        BoundInputAt history consumer (.result resultEvent .coefficient) bound actualMagnitude
    | resultSummary {consumer : Owner} {resultEvent : Nat} {resultOwner : Owner}
        {rawTerms : Option (List Term)} {bound : CoeffClass} {actualMagnitude : Nat}
        (scope : resultOwner.scope = consumer.scope)
        (projected : ProjectedBoundAt history resultEvent resultOwner rawTerms .summary
          bound actualMagnitude) :
        BoundInputAt history consumer (.result resultEvent .summary) bound actualMagnitude
    | predecessor {consumer : Owner} {inputPosition bindingEvent : Nat}
        {predecessor : ExpressionRef}
        {resultEvent frameStart : Nat} {rawTerms : Option (List Term)}
        {bound : CoeffClass} {actualMagnitude : Nat}
        (bindingRow : history.lookup bindingEvent = some
          ⟨.predecessor consumer inputPosition predecessor resultEvent, frameStart⟩)
        (projected : ProjectedBoundAt history resultEvent ⟨consumer.scope, predecessor⟩
          rawTerms .coefficient bound actualMagnitude) :
        BoundInputAt history consumer
          (.predecessor inputPosition bindingEvent .coefficient) bound actualMagnitude
    | transfer {consumer : Owner} {transferEvent transferFrame : Nat} {rule : BoundRule}
        {bound : CoeffClass} {actualMagnitude : Nat}
        (derived : BoundDerivedAt history transferEvent transferFrame consumer rule bound
          actualMagnitude) :
        BoundInputAt history consumer (.transfer transferEvent) bound actualMagnitude

  inductive BoundInputsAt (history : EventHistory) :
      Owner → List ValueRef → List CoeffClass → List Nat → Prop where
    | nil {consumer : Owner} : BoundInputsAt history consumer [] [] []
    | cons {consumer : Owner} {reference : ValueRef} {references : List ValueRef}
        {bound : CoeffClass} {bounds : List CoeffClass}
        {actualMagnitude : Nat} {actualMagnitudes : List Nat}
        (head : BoundInputAt history consumer reference bound actualMagnitude)
        (tail : BoundInputsAt history consumer references bounds actualMagnitudes) :
        BoundInputsAt history consumer (reference :: references) (bound :: bounds)
          (actualMagnitude :: actualMagnitudes)

  inductive FactorInputAt (history : EventHistory) :
      Owner → FactorEvidence → CoeffClass → Nat → Prop where
    | intro {consumer : Owner} {factor : FactorEvidence} {bound : CoeffClass}
        {actualMagnitude : Nat}
        (input : BoundInputAt history consumer factor.bound bound actualMagnitude) :
        FactorInputAt history consumer factor bound actualMagnitude

  inductive FactorInputsAt (history : EventHistory) :
      Owner → List FactorEvidence → List CoeffClass → List Nat → Prop where
    | nil {consumer : Owner} : FactorInputsAt history consumer [] [] []
    | cons {consumer : Owner} {factor : FactorEvidence} {factors : List FactorEvidence}
        {bound : CoeffClass} {bounds : List CoeffClass}
        {actualMagnitude : Nat} {actualMagnitudes : List Nat}
        (head : FactorInputAt history consumer factor bound actualMagnitude)
        (tail : FactorInputsAt history consumer factors bounds actualMagnitudes) :
        FactorInputsAt history consumer (factor :: factors) (bound :: bounds)
          (actualMagnitude :: actualMagnitudes)

  inductive BoundDerivedAt (history : EventHistory) :
      Nat → Nat → Owner → BoundRule → CoeffClass → Nat → Prop where
    | authority {producerEvent resultEvent frameStart : Nat} {owner : Owner}
        {authority : Authority} {bound : CoeffClass}
        (witness : AuthorityWitness history)
        (leaf : AuthorityLeafAt history producerEvent resultEvent frameStart owner authority bound) :
        BoundDerivedAt history producerEvent frameStart owner (.authority authority) bound
          (witness.authorityMagnitude resultEvent)
    | identity {transferEvent transferFrame : Nat} {owner : Owner}
        {reference : ValueRef} {bound : CoeffClass} {actualMagnitude : Nat}
        (transferRow : history.lookup transferEvent = some
          ⟨.boundTransfer owner (.identity reference), transferFrame⟩)
        (child : BoundInputAt history owner reference bound actualMagnitude) :
        BoundDerivedAt history transferEvent transferFrame owner (.identity reference)
          bound actualMagnitude
    | sum {transferEvent transferFrame : Nat} {owner : Owner}
        {references : List ValueRef} {bounds : List CoeffClass} {actuals : List Nat}
        (transferRow : history.lookup transferEvent = some
          ⟨.boundTransfer owner (.sum references), transferFrame⟩)
        (children : BoundInputsAt history owner references bounds actuals) :
        BoundDerivedAt history transferEvent transferFrame owner (.sum references)
          (addKnownList bounds) actuals.sum
    | scaleMagnitude {transferEvent transferFrame : Nat} {owner : Owner}
        {reference : ValueRef} {factor : Nat}
        {bound : CoeffClass} {actualMagnitude : Nat}
        (transferRow : history.lookup transferEvent = some
          ⟨.boundTransfer owner (.scale reference (.magnitude factor)), transferFrame⟩)
        (child : BoundInputAt history owner reference bound actualMagnitude) :
        BoundDerivedAt history transferEvent transferFrame owner
          (.scale reference (.magnitude factor)) (scaleMagnitude factor bound)
          (factor * actualMagnitude)
    | scaleValue {transferEvent transferFrame : Nat} {owner : Owner}
        {valueReference scaleReference : ValueRef}
        {valueBound scaleBound : CoeffClass} {valueActual scaleActual : Nat}
        (transferRow : history.lookup transferEvent = some
          ⟨.boundTransfer owner (.scale valueReference (.value scaleReference)), transferFrame⟩)
        (valueChild : BoundInputAt history owner valueReference valueBound valueActual)
        (scaleChild : BoundInputAt history owner scaleReference scaleBound scaleActual) :
        BoundDerivedAt history transferEvent transferFrame owner
          (.scale valueReference (.value scaleReference)) (scaleValue valueBound scaleBound)
          (valueActual * scaleActual)
    | monomialProduct {transferEvent transferFrame : Nat} {owner : Owner}
        {monomial : Monomial} {headFactor : FactorEvidence}
        {tailFactors : List FactorEvidence} {headBound : CoeffClass} {headActual : Nat}
        {tailBounds : List CoeffClass} {tailActuals : List Nat}
        (transferRow : history.lookup transferEvent = some
          ⟨.boundTransfer owner (.monomialProduct monomial (headFactor :: tailFactors)),
            transferFrame⟩)
        (head : FactorInputAt history owner headFactor headBound headActual)
        (tail : FactorInputsAt history owner tailFactors tailBounds tailActuals) :
        BoundDerivedAt history transferEvent transferFrame owner
          (.monomialProduct monomial (headFactor :: tailFactors))
          (productNonempty headBound tailBounds) (headActual * tailActuals.prod)
    | product {transferEvent transferFrame : Nat} {owner : Owner}
        {left right : ValueRef} {facts : TallSecurity0ABI.ProductFacts}
        {leftRows leftColumns rightRows rightColumns ringDimension factor : Nat}
        {leftBound rightBound : CoeffClass} {leftActual rightActual : Nat}
        (transferRow : history.lookup transferEvent = some
          ⟨.boundTransfer owner (.product left right facts), transferFrame⟩)
        (factorExact : EventReplay.productFactor leftRows leftColumns rightRows rightColumns
          ringDimension (replayProductFacts facts) = some factor)
        (leftChild : BoundInputAt history owner left leftBound leftActual)
        (rightChild : BoundInputAt history owner right rightBound rightActual) :
        BoundDerivedAt history transferEvent transferFrame owner (.product left right facts)
          (productWithFactor factor leftBound rightBound)
          (factor * leftActual * rightActual)
end

theorem ProjectedBoundAt.sound {history : EventHistory} {resultEvent : Nat} {owner : Owner}
    {rawTerms : Option (List Term)} {projection : Projection} {bound : CoeffClass}
    {actualMagnitude : Nat}
    (projected : ProjectedBoundAt history resultEvent owner rawTerms projection bound
      actualMagnitude) : bound.Interprets actualMagnitude := by
  refine ProjectedBoundAt.rec
    (motive_1 := fun _ _ _ _ bound actual _ => bound.Interprets actual)
    (motive_2 := fun _ _ bound actual _ => bound.Interprets actual)
    (motive_3 := fun _ _ bounds actuals _ =>
      List.Forall₂ (fun childBound childActual => childBound.Interprets childActual)
        bounds actuals)
    (motive_4 := fun _ _ bound actual _ => bound.Interprets actual)
    (motive_5 := fun _ _ bounds actuals _ =>
      List.Forall₂ (fun childBound childActual => childBound.Interprets childActual)
        bounds actuals)
    (motive_6 := fun _ _ _ _ bound actual _ => bound.Interprets actual)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ projected
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · exact .nil
  · intros
    exact .cons (by assumption) (by assumption)
  · intros
    assumption
  · exact .nil
  · intros
    exact .cons (by assumption) (by assumption)
  · intros
    apply AuthorityWitness.authorityBound <;> assumption
  · intros
    assumption
  · intros
    apply addKnownList_sound
    assumption
  · intros
    apply scaleMagnitude_sound
    assumption
  · intros
    apply scaleValue_sound <;> assumption
  · intros
    apply productNonempty_sound <;> assumption
  · intros
    exact (productWithFacts_sound (by assumption) (by assumption) (by assumption)).2

theorem BoundDerivedAt.sound {history : EventHistory} {transferEvent transferFrame : Nat}
    {owner : Owner} {rule : BoundRule} {bound : CoeffClass} {actualMagnitude : Nat}
    (derived : BoundDerivedAt history transferEvent transferFrame owner rule bound
      actualMagnitude) : bound.Interprets actualMagnitude := by
  refine BoundDerivedAt.rec
    (motive_1 := fun _ _ _ _ bound actual _ => bound.Interprets actual)
    (motive_2 := fun _ _ bound actual _ => bound.Interprets actual)
    (motive_3 := fun _ _ bounds actuals _ =>
      List.Forall₂ (fun childBound childActual => childBound.Interprets childActual)
        bounds actuals)
    (motive_4 := fun _ _ bound actual _ => bound.Interprets actual)
    (motive_5 := fun _ _ bounds actuals _ =>
      List.Forall₂ (fun childBound childActual => childBound.Interprets childActual)
        bounds actuals)
    (motive_6 := fun _ _ _ _ bound actual _ => bound.Interprets actual)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ derived
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · intros
    assumption
  · exact .nil
  · intros
    exact .cons (by assumption) (by assumption)
  · intros
    assumption
  · exact .nil
  · intros
    exact .cons (by assumption) (by assumption)
  · intros
    apply AuthorityWitness.authorityBound <;> assumption
  · intros
    assumption
  · intros
    apply addKnownList_sound
    assumption
  · intros
    apply scaleMagnitude_sound
    assumption
  · intros
    apply scaleValue_sound <;> assumption
  · intros
    apply productNonempty_sound <;> assumption
  · intros
    exact (productWithFacts_sound (by assumption) (by assumption) (by assumption)).2

theorem boundTransfer_to_resultCoefficient
    {history : EventHistory} {producerEvent resultEvent frameStart : Nat} {owner : Owner}
    {rule : BoundRule} {bound : CoeffClass} {actualMagnitude : Nat}
    (adjacent : producerEvent + 1 = resultEvent)
    (row : history.lookup resultEvent = some
      ⟨.resultCoefficient owner (coeffClassToTallBound bound), frameStart⟩)
    (derived : BoundDerivedAt history producerEvent frameStart owner rule bound
      actualMagnitude) :
    ProjectedBoundAt history resultEvent owner none .coefficient bound actualMagnitude :=
  .resultCoefficient adjacent row derived

theorem boundTransfer_to_resultExactCoefficient
    {history : EventHistory} {resultEvent coefficientProducer frameStart : Nat}
    {owner : Owner} {terms : List Term} {summary : Bound}
    {summaryProducer : Option Nat} {rule : BoundRule}
    {recordedCoefficientBound : Bound} {producerBound : CoeffClass}
    {actualMagnitude : Nat}
    (row : history.lookup resultEvent = some
      ⟨.resultExact owner terms recordedCoefficientBound coefficientProducer summary
        summaryProducer, frameStart⟩)
    (refines : RecordedBoundRefines recordedCoefficientBound producerBound)
    (derived : BoundDerivedAt history coefficientProducer frameStart owner rule producerBound
      actualMagnitude) :
    ProjectedBoundAt history resultEvent owner (some terms) .coefficient producerBound
      actualMagnitude :=
  .resultExactCoefficient row refines derived

theorem boundTransfer_to_resultExactSummary
    {history : EventHistory} {producerEvent resultEvent frameStart coefficientProducer : Nat}
    {owner : Owner} {terms : List Term} {recordedCoefficientBound : Bound}
    {rule : BoundRule} {producerBound : CoeffClass} {actualMagnitude : Nat}
    (row : history.lookup resultEvent = some
      ⟨.resultExact owner terms recordedCoefficientBound coefficientProducer
        (coeffClassToTallBound producerBound) (some producerEvent), frameStart⟩)
    (derived : BoundDerivedAt history producerEvent frameStart owner rule producerBound
      actualMagnitude) :
    ProjectedBoundAt history resultEvent owner (some terms) .summary producerBound
      actualMagnitude :=
  .resultExactSummary row derived

theorem boundDerived_to_transferInput
    {history : EventHistory} {transferEvent transferFrame : Nat} {owner : Owner}
    {rule : BoundRule} {bound : CoeffClass} {actualMagnitude : Nat}
    (derived : BoundDerivedAt history transferEvent transferFrame owner rule bound
      actualMagnitude) :
    BoundInputAt history owner (.transfer transferEvent) bound actualMagnitude :=
  .transfer derived

theorem projectedSummary_to_resultInput
    {history : EventHistory} {consumer : Owner} {resultEvent : Nat} {resultOwner : Owner}
    {rawTerms : Option (List Term)} {bound : CoeffClass} {actualMagnitude : Nat}
    (scope : resultOwner.scope = consumer.scope)
    (projected : ProjectedBoundAt history resultEvent resultOwner rawTerms .summary bound
      actualMagnitude) :
    BoundInputAt history consumer (.result resultEvent .summary) bound actualMagnitude :=
  .resultSummary scope projected

theorem centeredNorm_eq_zero_mod {modulus : Nat} {value : Int}
    (modulusPositive : 0 < modulus) (normZero : centeredNorm modulus value = 0) :
    value % Int.ofNat modulus = 0 := by
  unfold centeredNorm centeredCoefficient at normZero
  simp only [Int.natAbs_eq_zero] at normZero
  split at normZero
  · omega
  · split at normZero
    · have remainderNonnegative := Int.emod_nonneg value
        (Int.ofNat_ne_zero.mpr (Nat.ne_of_gt modulusPositive))
      have remainderLess := Int.emod_lt value
        (Int.ofNat_ne_zero.mpr (Nat.ne_of_gt modulusPositive))
      change value % Int.ofNat modulus < Int.ofNat modulus at remainderLess
      omega
    · have remainderNonnegative := Int.emod_nonneg value
        (Int.ofNat_ne_zero.mpr (Nat.ne_of_gt modulusPositive))
      have modulusNonnegative : (0 : Int) ≤ Int.ofNat modulus := by omega
      have remainderLess := Int.emod_lt value
        (Int.ofNat_ne_zero.mpr (Nat.ne_of_gt modulusPositive))
      change value % Int.ofNat modulus < Int.ofNat modulus at remainderLess
      omega

theorem exactClaim_mod_zero {Factor : Type} (modulus : Nat) (env : Env Factor)
    (actual : Int) (terms : Polynomial Factor)
    (claim : ValueClaim.Interprets modulus env actual (.exact terms .exactZero))
    (modulusPositive : 0 < modulus) :
    (actual - evalPolynomial env terms) % Int.ofNat modulus = 0 := by
  rcases claim with ⟨remainder, congruence, remainderBound⟩
  rw [congruence, centeredNorm_eq_zero_mod modulusPositive remainderBound]

/- A result event may change its polynomial representation while preserving its evaluated
   residue.  This bridge deliberately takes that residue equality as an explicit premise: the
   generator must supply the event-level canonical agreement, and Lean checks the consequence. -/
theorem exactValueClaim_of_eval_mod_zero {Factor : Type} (modulus : Nat) (env : Env Factor)
    (actual : Int) (input output : Polynomial Factor)
    (claim : ValueClaim.Interprets modulus env actual (.exact input .exactZero))
    (outputEvalMod : evalPolynomial env output % Int.ofNat modulus =
      evalPolynomial env input % Int.ofNat modulus)
    (modulusPositive : 0 < modulus) :
    ValueClaim.Interprets modulus env actual (.exact output .exactZero) := by
  have inputModZero := exactClaim_mod_zero modulus env actual input claim modulusPositive
  refine ⟨0, ?_, ?_⟩
  · have difference :
        (actual - evalPolynomial env output) % Int.ofNat modulus =
          (actual - evalPolynomial env input) % Int.ofNat modulus := by
      rw [Int.sub_emod, Int.sub_emod, outputEvalMod]
      simpa only [Int.emod_emod] using
        (Int.sub_emod actual (evalPolynomial env input) (Int.ofNat modulus)).symm
    rw [difference, inputModZero]
    simp
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

theorem exactValueClaim_add_of_mod_zero (modulus : Nat) (env : Env Owner)
    (leftActual rightActual : Int) (left right output : Polynomial Owner)
    (leftClaim : ValueClaim.Interprets modulus env leftActual (.exact left .exactZero))
    (rightClaim : ValueClaim.Interprets modulus env rightActual (.exact right .exactZero))
    (outputEval : evalPolynomial env output =
      evalPolynomial env left + evalPolynomial env right)
    (modulusPositive : 0 < modulus) :
    ValueClaim.Interprets modulus env (leftActual + rightActual) (.exact output .exactZero) := by
  refine ⟨0, ?_, ?_⟩
  · have leftModZero := exactClaim_mod_zero modulus env leftActual left leftClaim modulusPositive
    have rightModZero := exactClaim_mod_zero modulus env rightActual right rightClaim modulusPositive
    rw [outputEval]
    calc
      (leftActual + rightActual - (evalPolynomial env left + evalPolynomial env right)) %
          Int.ofNat modulus =
          ((leftActual - evalPolynomial env left) +
            (rightActual - evalPolynomial env right)) % Int.ofNat modulus := by
              congr 1 <;> omega
      _ = ((leftActual - evalPolynomial env left) % Int.ofNat modulus +
        (rightActual - evalPolynomial env right) % Int.ofNat modulus) % Int.ofNat modulus := by
          rw [Int.add_emod]
      _ = 0 := by rw [leftModZero, rightModZero]; simp
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

/-- Adding an exact-zero remainder preserves the finite remainder carried by the left claim. -/
theorem exactValueClaim_add_right_mod_zero {Factor : Type} (modulus : Nat) (env : Env Factor)
    (leftActual rightActual : Int) (leftTerms rightTerms output : Polynomial Factor)
    (maximum : Nat)
    (leftClaim : ValueClaim.Interprets modulus env leftActual
      (.exact leftTerms (.finite maximum)))
    (rightModZero : (rightActual - evalPolynomial env rightTerms) % Int.ofNat modulus = 0)
    (outputEval : evalPolynomial env output =
      evalPolynomial env leftTerms + evalPolynomial env rightTerms) :
    ValueClaim.Interprets modulus env (leftActual + rightActual)
      (.exact output (.finite maximum)) := by
  rcases leftClaim with ⟨remainder, leftCongruence, leftBound⟩
  refine ⟨remainder, ?_, leftBound⟩
  rw [outputEval]
  calc
    (leftActual + rightActual -
        (evalPolynomial env leftTerms + evalPolynomial env rightTerms)) %
          Int.ofNat modulus =
        ((leftActual - evalPolynomial env leftTerms) +
          (rightActual - evalPolynomial env rightTerms)) % Int.ofNat modulus := by
            congr 1 <;> omega
    _ = ((leftActual - evalPolynomial env leftTerms) % Int.ofNat modulus +
      (rightActual - evalPolynomial env rightTerms) % Int.ofNat modulus) %
        Int.ofNat modulus := by
          rw [Int.add_emod]
    _ = (leftActual - evalPolynomial env leftTerms) % Int.ofNat modulus := by
      rw [rightModZero]
      simp
    _ = remainder % Int.ofNat modulus := leftCongruence

theorem exactValueClaim_sub_exactZero_of_mod_zero (modulus : Nat) (env : Env Owner)
    (leftActual rightActual : Int) (left right output : Polynomial Owner)
    (leftClaim : ValueClaim.Interprets modulus env leftActual (.exact left .exactZero))
    (rightClaim : ValueClaim.Interprets modulus env rightActual (.exact right .exactZero))
    (outputEval : evalPolynomial env output =
      evalPolynomial env left - evalPolynomial env right)
    (modulusPositive : 0 < modulus) :
    ValueClaim.Interprets modulus env (leftActual - rightActual) (.exact output .exactZero) := by
  refine ⟨0, ?_, ?_⟩
  · have leftModZero := exactClaim_mod_zero modulus env leftActual left leftClaim modulusPositive
    have rightModZero := exactClaim_mod_zero modulus env rightActual right rightClaim modulusPositive
    rw [outputEval]
    calc
      (leftActual - rightActual - (evalPolynomial env left - evalPolynomial env right)) %
          Int.ofNat modulus =
          ((leftActual - evalPolynomial env left) -
            (rightActual - evalPolynomial env right)) % Int.ofNat modulus := by
              congr 1 <;> omega
      _ = ((leftActual - evalPolynomial env left) % Int.ofNat modulus -
        (rightActual - evalPolynomial env right) % Int.ofNat modulus) % Int.ofNat modulus := by
          rw [Int.sub_emod]
      _ = 0 := by rw [leftModZero, rightModZero]; simp
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

theorem exactValueClaim_product_of_mod_zero (modulus : Nat) (env : Env Owner)
    (leftActual rightActual : Int) (left right output : Polynomial Owner)
    (leftClaim : ValueClaim.Interprets modulus env leftActual (.exact left .exactZero))
    (rightClaim : ValueClaim.Interprets modulus env rightActual (.exact right .exactZero))
    (outputEval : evalPolynomial env output =
      evalPolynomial env left * evalPolynomial env right)
    (modulusPositive : 0 < modulus) :
    ValueClaim.Interprets modulus env (leftActual * rightActual) (.exact output .exactZero) := by
  have leftModZero := exactClaim_mod_zero modulus env leftActual left leftClaim modulusPositive
  have rightModZero := exactClaim_mod_zero modulus env rightActual right rightClaim modulusPositive
  have leftResidue : leftActual % Int.ofNat modulus = evalPolynomial env left % Int.ofNat modulus :=
    Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr leftModZero
  have rightResidue : rightActual % Int.ofNat modulus =
      evalPolynomial env right % Int.ofNat modulus :=
    Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr rightModZero
  refine ⟨0, ?_, ?_⟩
  · rw [outputEval]
    have productResidue :
        (leftActual * rightActual) % Int.ofNat modulus =
          (evalPolynomial env left * evalPolynomial env right) % Int.ofNat modulus := by
      rw [Int.mul_emod, Int.mul_emod, leftResidue, rightResidue]
      simpa only [Int.emod_emod] using
        (Int.mul_emod (evalPolynomial env left) (evalPolynomial env right)
          (Int.ofNat modulus)).symm
    exact Int.emod_eq_emod_iff_emod_sub_eq_zero.mp productResidue
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

theorem exactValueClaim_sub_of_mod_zero {Factor : Type} (modulus : Nat) (env : Env Factor)
    (leftActual rightActual : Int) (leftTerms rightTerms output : Polynomial Factor)
    (maximum : Nat)
    (leftClaim : ValueClaim.Interprets modulus env leftActual (.exact leftTerms (.finite maximum)))
    (rightModZero : (rightActual - evalPolynomial env rightTerms) % Int.ofNat modulus = 0)
    (outputEval : evalPolynomial env output = evalPolynomial env leftTerms - evalPolynomial env rightTerms) :
    ValueClaim.Interprets modulus env (leftActual - rightActual)
      (.exact output (.finite maximum)) := by
  rcases leftClaim with ⟨remainder, leftCongruence, leftBound⟩
  refine ⟨remainder, ?_, leftBound⟩
  rw [outputEval]
  calc
    (leftActual - rightActual - (evalPolynomial env leftTerms - evalPolynomial env rightTerms)) %
        Int.ofNat modulus =
        ((leftActual - evalPolynomial env leftTerms) -
          (rightActual - evalPolynomial env rightTerms)) % Int.ofNat modulus := by
            congr 1 <;> omega
    _ = ((leftActual - evalPolynomial env leftTerms) % Int.ofNat modulus -
      (rightActual - evalPolynomial env rightTerms) % Int.ofNat modulus) % Int.ofNat modulus := by
        rw [Int.sub_emod]
    _ = (leftActual - evalPolynomial env leftTerms) % Int.ofNat modulus := by
      rw [rightModZero]
      simp
    _ = remainder % Int.ofNat modulus := by simpa [boundInterprets] using leftCongruence

theorem centeredNorm_eq_of_emod_eq {modulus : Nat} {left right : Int}
    (modulusPositive : 0 < modulus)
    (residueEqual : left % Int.ofNat modulus = right % Int.ofNat modulus) :
    centeredNorm modulus left = centeredNorm modulus right := by
  unfold centeredNorm centeredCoefficient
  simp only [Nat.ne_of_gt modulusPositive, ↓reduceIte]
  rw [residueEqual]

theorem centeredNorm_le_of_empty_finite_claim {Factor : Type} (modulus : Nat)
    (env : Env Factor) (actual : Int) (maximum : Nat)
    (claim : ValueClaim.Interprets modulus env actual (.exact [] (.finite maximum)))
    (modulusPositive : 0 < modulus) :
    centeredNorm modulus actual ≤ maximum := by
  rcases claim with ⟨remainder, congruence, remainderBound⟩
  have residueEqual : actual % Int.ofNat modulus = remainder % Int.ofNat modulus := by
    simpa [evalPolynomial] using congruence
  rw [centeredNorm_eq_of_emod_eq modulusPositive residueEqual]
  simpa [boundInterprets] using remainderBound

theorem finalStrictBound_of_empty_finite_claim {Factor : Type} (plaintextModulus modulus : Nat)
    (env : Env Factor) (actual : Int) (maximum : Nat)
    (claim : ValueClaim.Interprets modulus env actual (.exact [] (.finite maximum)))
    (modulusPositive : 0 < modulus)
    (certificateBound : 2 * plaintextModulus * maximum < modulus) :
    2 * plaintextModulus * centeredNorm modulus actual < modulus := by
  have actualBound :=
    centeredNorm_le_of_empty_finite_claim modulus env actual maximum claim modulusPositive
  exact Nat.lt_of_le_of_lt (Nat.mul_le_mul_left (2 * plaintextModulus) actualBound)
    certificateBound

theorem invocationEndSound (modulus : Nat) (env : Env Owner) (actual : Int)
    (prefoldTerms endTerms : Polynomial Owner)
    (prefoldSummary endSummary : Bound)
    (claim : ValueClaim.Interprets modulus env actual (.exact prefoldTerms prefoldSummary))
    (termsExact : endTerms = prefoldTerms)
    (summaryExact : endSummary = prefoldSummary) :
    ValueClaim.Interprets modulus env actual (.exact endTerms endSummary) := by
  simpa [termsExact, summaryExact] using claim

/-- A finite raw bound is supplied constructively; no decimal parser belongs to this layer. -/
def rawCoefficientClassInterprets (modulus : Nat) (value : Int) :
    RawCoefficientClass → Prop
  | .exactZero => centeredNorm modulus value = 0
  | .finite encoded =>
      ∃ maximum : Nat, encoded = toString maximum ∧ centeredNorm modulus value ≤ maximum
  | .large => True

def rawValueContractInterprets (modulus : Nat) (value : Int)
    (contract : RawValueContract) : Prop :=
  ∃ coefficientClass,
    contract.coefficientClass = some coefficientClass ∧
      rawCoefficientClassInterprets modulus value coefficientClass

def ownerAtSelector (document : TallDocument) (selector : Option Nat) (owner : Owner) : Prop :=
  match document.residualRoot, owner.scope, selector with
  | .closed root, .closed ownerRoot, none => root = ownerRoot
  | .family program domain, .program ownerProgram, some selected =>
      program = ownerProgram ∧ domain.minimum ≤ selected ∧ selected < domain.maximumExclusive
  | _, _, _ => False

def sourceRef? : ExpressionDescriptor → Option SourceRef
  | .source (.direct source) | .source (.family source _) => some source
  | _ => none

def sourceContract? : SchemaV1.SourceRow → Option RawValueContract
  | .direct _ _ contract | .family _ contract => contract
  | .constant _ => none

def expressionEventRefs : ExpressionDescriptor → List EventRef
  | .event (.sample event) | .event (.sampler event) => [event]
  | .event (.gadgetDecompose events) => events
  | .operation (.event (.sample event)) _ | .operation (.event (.sampler event)) _ => [event]
  | .operation (.event (.gadgetDecompose events)) _ => events
  | _ => []

def eventContract? : SchemaV1.EventRow → Option RawValueContract
  | .sample _ _ contract | .sampler _ _ contract | .gadgetDecompose _ _ _ _ _ _ _ contract =>
      contract

def sourceRawContract? (document : TallDocument) (source : SourceRef) :
    Option RawValueContract :=
  (document.sources.lookup source.row).bind sourceContract?

def eventRawContract? (document : TallDocument) (event : EventRef) : Option RawValueContract :=
  (document.events.lookup event.row).bind eventContract?

def monomialContains (monomial : Monomial) (owner : Owner) : Prop :=
  owner ∈ monomial.centralFactors ∨ owner ∈ monomial.orderedFactors

def termContains (term : Term) (owner : Owner) : Prop := monomialContains term.monomial owner

def eventContainsFactor (event : Event) (owner : Owner) : Prop :=
  match event with
  | .resultExact _ terms _ _ _ _ | .invocationEndExact _ _ terms _ _ _ _ |
      .preFoldPolynomial _ terms _ _ => ∃ term ∈ terms, termContains term owner
  | .appliedRelation _ sourceMonomial _ _ _ rule =>
      monomialContains sourceMonomial owner ∨ match rule with
        | .universal _ lhs _ _ => monomialContains lhs owner
        | .gadget gadget decomposition _ _ => gadget = owner ∨ decomposition = owner
  | .boundTransfer _ (.monomialProduct monomial _) => monomialContains monomial owner
  | .coefficientMerge merge => monomialContains merge.output owner
  | _ => False

def FactorAtomAt (history : EventHistory) (event : Nat) (owner : Owner) : Prop :=
  ∃ entry, history.lookup event = some entry ∧ eventContainsFactor entry.event owner

def SourceFactorAt (document : TallDocument) (history : EventHistory) (selector : Option Nat)
    (owner : Owner) (factorEvent : Nat) (source : SourceRef) : Prop :=
  FactorAtomAt history factorEvent owner ∧ ownerAtSelector document selector owner ∧
    ∃ expression sourceRow,
      document.expressions.lookup owner.expression.row = some expression ∧
        sourceRef? expression.descriptor = some source ∧
        document.sources.lookup source.row = some sourceRow

def SamplerFactorAt (document : TallDocument) (history : EventHistory) (selector : Option Nat)
    (owner : Owner) (factorEvent : Nat) (event : EventRef) : Prop :=
  FactorAtomAt history factorEvent owner ∧ ownerAtSelector document selector owner ∧
    ∃ expression eventRow,
      document.expressions.lookup owner.expression.row = some expression ∧
        event ∈ expressionEventRefs expression.descriptor ∧
        document.events.lookup event.row = some eventRow

def RelationApplicationAt (document : TallDocument) (history : EventHistory)
    (selector : Option Nat) (application : Nat) : Prop :=
  ∃ entry owner sourceMonomial outerCoefficient orderedStart orderedEndExclusive rule,
    history.lookup application = some entry ∧
      entry.event = .appliedRelation owner sourceMonomial outerCoefficient orderedStart
        orderedEndExclusive rule ∧
      ownerAtSelector document selector owner ∧
      TallSecurity0ABI.ownerValid document owner = true

def exactTermsAt? (history : EventHistory) (event : Nat) : Option (List Term) :=
  match TallSecurity0ABI.eventAt? history event with
  | some (.resultExact _ terms _ _ _ _) |
      some (.invocationEndExact _ _ terms _ _ _ _) => some terms
  | _ => none

def RelationCongruent (modulus : Nat) (history : EventHistory) (env : Env Owner)
    (application : Nat) : Prop :=
  match TallSecurity0ABI.eventAt? history application with
  | some (.appliedRelation _ _ _ _ _ (.universal computed lhs _ rhsResult)) =>
      (∃ computedOwner dispatch source,
        TallSecurity0ABI.eventAt? history computed =
            some (Event.specializationComputed computedOwner dispatch source) ∧
          source.end = computed ∧ source.start ≤ rhsResult ∧ rhsResult < source.end) ∧
        ∃ rhsTerms,
          exactTermsAt? history rhsResult = some rhsTerms ∧
            evalMonomial env lhs.toKey % Int.ofNat modulus =
              evalPolynomial env (rhsTerms.map Term.toExact) % Int.ofNat modulus
  | some (.appliedRelation _ _ _ _ _ (.gadget gadget decomposition _ inputResult)) =>
      ∃ inputTerms,
        exactTermsAt? history inputResult = some inputTerms ∧
          (env gadget * env decomposition) % Int.ofNat modulus =
            evalPolynomial env (inputTerms.map Term.toExact) % Int.ofNat modulus
  | _ => False

def canonicalSelfTerm (owner : Owner) : Term :=
  { monomial := { centralFactors := [], orderedFactors := [owner] }
    coefficient := 1 }

inductive ReachedTerminalRule : BoundRule → Prop where
  | authorityFactStore : ReachedTerminalRule (.authority .factStore)
  | authorityProgramFamilyFact : ReachedTerminalRule (.authority .programFamilyFact)
  | authorityOperator : ReachedTerminalRule (.authority .operator)
  | authorityRelationPreimageSource (source : ExpressionRef) :
      ReachedTerminalRule (.authority (.relationPreimageSource source))
  | identity (input : ValueRef) : ReachedTerminalRule (.identity input)
  | scale (value : ValueRef) (factor : Scale) : ReachedTerminalRule (.scale value factor)

/-- An interpreted exact claim tied to the precise `Result` row at one history index. -/
structure ExactClaimAt (history : EventHistory) (modulus : Nat) (env : Env Owner)
    (event : Nat) (owner : Owner) (actual : Int) (rawTerms : List Term)
    (summary : Bound) : Prop where
  row : ∃ coefficientBound coefficientProducer summaryProducer,
    (history.lookup event).map AnnotatedEvent.event =
      some (.resultExact owner rawTerms coefficientBound coefficientProducer summary
        summaryProducer)
  claim : ValueClaim.Interprets modulus env actual
    (.exact (rawTerms.map Term.toExact) summary)

/-! Exact finite claims require both sides of the semantic contract: the concrete ResultExact
row and an explicit Rust-to-Lean congruence plus remainder bound. -/
theorem exactFiniteClaimAt
    {history : EventHistory} {modulus resultEvent : Nat} {env : Env Owner}
    {owner : Owner} {rawTerms : List Term} {maximum : Nat} {actual remainder : Int}
    {frameStart coefficientProducer : Nat} {coefficientBound : Bound}
    {summaryProducer : Option Nat}
    (row : history.lookup resultEvent = some
      ⟨.resultExact owner rawTerms coefficientBound coefficientProducer (.finite maximum)
        summaryProducer, frameStart⟩)
    (congruence :
      (actual - evalPolynomial env (rawTerms.map Term.toExact)) % Int.ofNat modulus =
        remainder % Int.ofNat modulus)
    (remainderLe : centeredNorm modulus remainder ≤ maximum) :
    ExactClaimAt history modulus env resultEvent owner actual rawTerms (.finite maximum) := by
  refine ⟨⟨coefficientBound, coefficientProducer, summaryProducer, ?_⟩,
    exactValueClaim_of_remainder modulus env actual
    (rawTerms.map Term.toExact) maximum remainder congruence remainderLe⟩
  rw [row]
  rfl

def TerminalExactAt (document : TallDocument) (history : EventHistory)
    (selector : Option Nat) (producer resultEvent : Nat) (owner : Owner)
    (rawTerms : List Term) : Prop :=
  producer + 1 = resultEvent ∧ ownerAtSelector document selector owner ∧
    ∃ rule frameStart coefficientBound,
      ReachedTerminalRule rule ∧
        history.lookup producer = some ⟨.boundTransfer owner rule, frameStart⟩ ∧
        history.lookup resultEvent = some
          ⟨.resultExact owner rawTerms coefficientBound producer .exactZero none, frameStart⟩

/-- Honest primitive contracts and all reached relation congruences for one selector. -/
structure Witness (document : TallDocument) (history : EventHistory) (selector : Option Nat)
    (modulus : Nat) extends AuthorityWitness history where
  env : Env Owner
  sourceBound : ∀ owner factorEvent source,
    SourceFactorAt document history selector owner factorEvent source →
      ∃ contract, sourceRawContract? document source = some contract ∧
        rawValueContractInterprets modulus (env owner) contract
  samplerBound : ∀ owner factorEvent event,
    SamplerFactorAt document history selector owner factorEvent event →
      ∃ contract, eventRawContract? document event = some contract ∧
        rawValueContractInterprets modulus (env owner) contract
  relationCongruence : ∀ application,
    RelationApplicationAt document history selector application →
      RelationCongruent modulus history env application
  honestTerminalActual : Nat → Int
  honestTerminalCongruence : ∀ producer resultEvent owner rawTerms,
    TerminalExactAt document history selector producer resultEvent owner rawTerms →
      (honestTerminalActual resultEvent - evalPolynomial env (rawTerms.map Term.toExact)) %
        Int.ofNat modulus = 0

/-! Atomic derivations require validated source or sampler provenance.  The result indices are
    intentional: constructors cannot choose an arbitrary actual value, term list, summary, or
    claim. -/

def canonicalSelfClaim (owner : Owner) : ValueClaim Owner :=
  .exact [canonicalSelfTerm owner |>.toExact] .exactZero

def exactResultAt (history : EventHistory) (resultEvent : Nat) (owner : Owner) : Prop :=
  ∃ frameStart coefficientBound coefficientProducer,
    history.lookup resultEvent =
      some ⟨.resultExact owner [canonicalSelfTerm owner] coefficientBound coefficientProducer
        .exactZero none, frameStart⟩

theorem terminalExactClaimAt {document : TallDocument} {history : EventHistory}
    {selector : Option Nat} {modulus producer resultEvent : Nat} {owner : Owner}
    {rawTerms : List Term}
    (witness : Witness document history selector modulus)
    (terminal : TerminalExactAt document history selector producer resultEvent owner rawTerms) :
    ExactClaimAt history modulus witness.env resultEvent owner
      (witness.honestTerminalActual resultEvent) rawTerms .exactZero := by
  refine ⟨?_, 0, ?_, ?_⟩
  · rcases terminal with ⟨_, _, rule, frameStart, coefficientBound, _, _, resultAt⟩
    refine ⟨coefficientBound, producer, none, ?_⟩
    rw [resultAt]
    rfl
  · simpa using
      witness.honestTerminalCongruence producer resultEvent owner rawTerms terminal
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

inductive ValueDerived (document : TallDocument) (history : EventHistory)
    (selector : Option Nat) (modulus : Nat)
    (witness : Witness document history selector modulus)
    (owner : Owner) (resultEvent : Nat) : Int → ValueClaim Owner → Prop where
  | sourceAtom (source : SourceRef)
      (result : exactResultAt history resultEvent owner)
      (factor : SourceFactorAt document history selector owner resultEvent source) :
      ValueDerived document history selector modulus witness owner resultEvent
        (witness.env owner) (canonicalSelfClaim owner)
  | samplerAtom (event : EventRef)
      (result : exactResultAt history resultEvent owner)
      (factor : SamplerFactorAt document history selector owner resultEvent event) :
      ValueDerived document history selector modulus witness owner resultEvent
        (witness.env owner) (canonicalSelfClaim owner)

abbrev DerivedResult (document : TallDocument) (history : EventHistory)
    (selector : Option Nat) (modulus : Nat)
    (witness : Witness document history selector modulus)
    (owner : Owner) (resultEvent : Nat) : Prop :=
  ValueDerived document history selector modulus witness owner resultEvent
    (witness.env owner) (canonicalSelfClaim owner)

theorem ValueDerived.interprets {document : TallDocument} {history : EventHistory}
    {selector : Option Nat} {modulus : Nat}
    {witness : Witness document history selector modulus}
    {owner : Owner} {resultEvent : Nat}
    (_ : DerivedResult document history selector modulus witness owner resultEvent) :
    ValueClaim.Interprets modulus witness.env (witness.env owner) (canonicalSelfClaim owner) := by
  refine ⟨0, ?_, ?_⟩
  · simp [evalPolynomial, evalMonomial, canonicalSelfTerm, Term.toExact, Monomial.toKey]
  · simp [boundInterprets, centeredNorm, centeredCoefficient]

abbrev TallEnv := Env Owner

abbrev TallValueClaim := ValueClaim Owner

def ForStatement (root : SchemaV1.ResidualRoot) (claim : Option Nat → Prop) : Prop :=
  match root with
  | .closed _ => claim none
  | .family _ domain =>
      ∀ selector, domain.minimum ≤ selector → selector < domain.maximumExclusive →
        claim (some selector)

def rootMatchesOwner (root : SchemaV1.ResidualRoot) (owner : Owner) : Prop :=
  match root, owner.scope with
  | .closed expression, .closed ownerExpression => expression = ownerExpression
  | .family program _, .program ownerProgram => program = ownerProgram
  | _, _ => False

/-- The fixed Security0 statement; later checkpoints construct, rather than assume, its proof. -/
def Security0Accepted (document : TallDocument) (history : EventHistory)
    (plaintextModulus ciphertextModulus ringDimension finalEvent preFoldEvent finalBound : Nat)
    (finalOwner : Owner) (finalTerms : List Term) (finalCoefficientBound : Bound)
    (finalCoefficientProducer : Nat) (finalSummary : Bound)
    (finalSummaryProducer : Option Nat)
    (residual : Option Nat → Env Owner → Int) : Prop :=
  TallSecurity0ABI.Valid document history ∧
    document.plaintextModulus = toString plaintextModulus ∧
    document.ciphertextModulus = toString ciphertextModulus ∧
    document.ringDimension = ringDimension ∧
    rootMatchesOwner document.residualRoot finalOwner ∧
    finalSummary = .finite finalBound ∧
    (∃ frameStart,
      history.lookup finalEvent = some
        ⟨.invocationEndExact finalOwner preFoldEvent finalTerms finalCoefficientBound
          finalCoefficientProducer finalSummary finalSummaryProducer, frameStart⟩) ∧
    ForStatement document.residualRoot fun selector ↦
      ∀ witness : Witness document history selector ciphertextModulus,
        ValueClaim.Interprets ciphertextModulus witness.env (residual selector witness.env)
            (.exact (finalTerms.map Term.toExact) finalSummary) ∧
          2 * plaintextModulus * centeredNorm ciphertextModulus
            (residual selector witness.env) < ciphertextModulus

theorem forall₂_append {α β : Type} {r : α → β → Prop}
    {left left' right right' : List _} :
    List.Forall₂ r left left' → List.Forall₂ r right right' →
      List.Forall₂ r (left ++ right) (left' ++ right') := by
  intro first second
  induction first generalizing right right' with
  | nil => simpa using second
  | cons head tail ih => exact List.Forall₂.cons head (ih second)

end Mxx.Certificate.OperationalNoise.TallSemantics
