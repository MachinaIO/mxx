import MxxIR.Rel

namespace MxxIR

universe u v w

/- A named scope specification is a proposition over the generated relation and its explicit
   input/output contracts.  It is not a second graph or certificate datatype. -/
def ScopeSpec {Config : Type u} {Inputs Outputs : Config → Type v}
    (run : (cfg : Config) → Rel (Inputs cfg) (Outputs cfg))
    (P : (cfg : Config) → Inputs cfg → Prop)
    (Q : (cfg : Config) → Inputs cfg → Outputs cfg → Prop) : Prop :=
  ∀ cfg inputs outputs, P cfg inputs → run cfg inputs outputs → Q cfg inputs outputs

theorem ScopeSpec.conclusion {Config : Type u} {Inputs Outputs : Config → Type v}
    {run : (cfg : Config) → Rel (Inputs cfg) (Outputs cfg)}
    {P : (cfg : Config) → Inputs cfg → Prop}
    {Q : (cfg : Config) → Inputs cfg → Outputs cfg → Prop}
    (h : ScopeSpec run P Q) {cfg : Config} {input : Inputs cfg} {output : Outputs cfg}
    (input_ok : P cfg input) (run_ok : run cfg input output) : Q cfg input output :=
  h cfg input output input_ok run_ok

end MxxIR
