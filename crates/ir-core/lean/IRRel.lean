universe u v w

namespace MxxIR

/- A relation between an input and an output.  Generated scopes use this type
   directly; there is no semantic environment or graph interpreter. -/
abbrev Rel (Input : Type u) (Output : Type v) := Input → Output → Prop

def Rel.comp {A : Type u} {B : Type v} {C : Type w} (left : Rel A B) (right : Rel B C) :
    Rel A C :=
  fun input output => ∃ middle, left input middle ∧ right middle output

def Rel.map {A : Type u} {B : Type v} {C : Type w} (f : A → B) (relation : Rel B C) :
    Rel A C :=
  fun input output => relation (f input) output

def Rel.comap {A : Type u} {B : Type v} {C : Type w} (relation : Rel A B) (f : C → A) :
    Rel C B :=
  fun input output => relation (f input) output

def pointwise {n : Nat} {Input : Fin n → Type u} {Output : Fin n → Type v}
    (body : (i : Fin n) → Rel (Input i) (Output i)) :
    Rel ((i : Fin n) → Input i) ((i : Fin n) → Output i) :=
  fun inputs outputs => ∀ i, body i (inputs i) (outputs i)

theorem pointwise_apply {n : Nat} {Input : Fin n → Type u} {Output : Fin n → Type v}
    {body : (i : Fin n) → Rel (Input i) (Output i)}
    {inputs : (i : Fin n) → Input i} {outputs : (i : Fin n) → Output i} (i : Fin n)
    (h : pointwise body inputs outputs) : body i (inputs i) (outputs i) :=
  h i

def broadcast {Input : Type u} {Output : Type v} (body : Rel Input Output) (n : Nat) :
    Rel Input (Fin n → Output) :=
  fun input outputs => ∀ i, body input (outputs i)

def zip {n : Nat} {Input : Type u} {Output : Type v} (body : Rel Input Output)
    (inputs : Fin n → Input) :
    Rel Unit (Fin n → Output) :=
  fun _ outputs => ∀ i, body (inputs i) (outputs i)

end MxxIR
