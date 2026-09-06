import ExponentProgram
import ExponentMonomial

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Expose the state index of a checked derivation when composing generated scopes. -/
def ProgramDerivation.state {q n secret columns bound : Nat}
    {gadget : ExactMatrix q n secret columns} {state : ProgramState q n secret columns}
    (_proof : ProgramDerivation gadget state bound) : ProgramState q n secret columns := state

/-- The native program stays in the strided subgroup declared by its inputs. -/
def StridedMonomial {q n : Nat} (stride width : Nat) (value : ExactPoly q n) : Prop :=
  ∃ exponent : Fin (2 * width), value =
    AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ (stride * exponent.val)

theorem StridedMonomial.signed {q n stride width : Nat} (hn : 0 < n)
    (hdimension : n = stride * width) {value : ExactPoly q n}
    (hvalue : StridedMonomial stride width value) : SignedMonomial value := by
  obtain ⟨exponent, equation⟩ := hvalue
  have hs : 0 < stride := by nlinarith
  refine ⟨⟨stride * exponent.val, ?_⟩, equation⟩
  have h := Nat.mul_lt_mul_of_pos_left exponent.isLt hs
  nlinarith

theorem StridedMonomial.mul {q n stride width : Nat} (hwidth : 0 < width)
    (hdimension : n = stride * width) {left right : ExactPoly q n}
    (hl : StridedMonomial stride width left) (hr : StridedMonomial stride width right) :
    StridedMonomial stride width (left * right) := by
  obtain ⟨a, ha⟩ := hl
  obtain ⟨b, hb⟩ := hr
  let root := AdjoinRoot.root (negacyclicModulus n (ZMod q))
  have hp : (root ^ stride) ^ (2 * width) = 1 := by
    rw [← pow_mul, show stride * (2 * width) = n * 2 by rw [hdimension]; ring]
    rw [pow_mul, Negacyclic.root_pow_n]
    ring
  have wrap (k : Nat) : (root ^ stride) ^ k = (root ^ stride) ^ (k % (2 * width)) := by
    conv_lhs => rw [← Nat.div_add_mod k (2 * width)]
    rw [pow_add, pow_mul, hp]
    simp
  refine ⟨⟨(a.val + b.val) % (2 * width), Nat.mod_lt _ (by omega)⟩, ?_⟩
  rw [ha, hb]
  simpa only [pow_mul, ← pow_add] using wrap (a.val + b.val)

/-- This is the evaluator's private-then-public message order, including zero branches. -/
theorem native_family_message {q n stride width count : Nat} (hwidth : 0 < width)
    (hdimension : n = stride * width) (message : ExactPoly q n)
    (selector publicMessage : Fin count → ExactPoly q n) (selected : Fin count)
    (hmessage : StridedMonomial stride width message)
    (hpublic : StridedMonomial stride width (publicMessage selected))
    (hone : selector selected = 1)
    (hzero : ∀ i, i ≠ selected → selector i = 0) :
    StridedMonomial stride width (∑ i, publicMessage i * (message * selector i)) := by
  rw [Finset.sum_eq_single selected]
  · simpa [hone] using hpublic.mul hwidth hdimension hmessage
  · intro i hi hne
    simp [hzero i hne]
  · simp

/-- Close one native bucket after the implementation's balanced addition tree. -/
theorem native_family_reduce {q n secret columns count stride width : Nat}
    (hwidth : 0 < width) (hdimension : n = stride * width)
    (gadget : ExactMatrix q n secret columns)
    (states : Fin count → ProgramState q n secret columns)
    (mask payload : ExactMatrix q n 1 secret) (bound : Nat)
    (previous : ∀ i, ProgramDerivation gadget (states i) bound)
    (sharedMask : ∀ i, (states i).mask = mask)
    (sharedPayload : ∀ i, (states i).payload = payload)
    (message : ExactPoly q n) (selector publicMessage : Fin count → ExactPoly q n)
    (selected : Fin count)
    (hmessage : StridedMonomial stride width message)
    (hpublic : StridedMonomial stride width (publicMessage selected))
    (hone : selector selected = 1) (hzero : ∀ i, i ≠ selected → selector i = 0)
    (messages : ∀ i, (states i).message = publicMessage i * (message * selector i)) :
    ProgramDerivation gadget (ProgramState.familyOutput states mask payload) (count * bound) ∧
      StridedMonomial stride width (ProgramState.familyOutput states mask payload).message := by
  constructor
  · simpa using ProgramDerivation.familyReduce states (fun _ => bound) mask payload
      previous sharedMask sharedPayload
  · change StridedMonomial stride width (∑ i, (states i).message)
    simp_rw [messages]
    exact native_family_message hwidth hdimension message selector publicMessage selected
      hmessage hpublic hone hzero

end MxxExponentLut
