import W_Ideal

/-!
# The ideal functionality

On the evaluator's chunks `x = (0, 1)` the function's two table steps give
`z_1 = [1, 0, 1, 0][z_0 + y_0]` and `z_2 = [0, 1, 1, 0][z_1 + 1 + y_1]`.
-/

namespace Dio2Proof

/-- The first table step. -/
def table0 : List Int := [1, 0, 1, 0]

/-- The second table step. -/
def table1 : List Int := [0, 1, 1, 0]

/-- Entry `index` of `table`, `0` outside it. -/
def entry (table : List Int) (index : Int) : Int := table.getD index.toNat 0

/-- `select index [a, b, c, d] out` gives the entry. -/
theorem select_entry {index out a b c d : Int}
    (h : MxxRuntime.select index [a, b, c, d] out) : out = entry [a, b, c, d] index := by
  obtain ⟨position, hposition, hout⟩ := h
  subst hposition
  fin_cases position <;> simp [hout, entry]

/-- The ideal output on `z_0` and the bits `y`. -/
theorem ideal_value (params : W_Ideal.Params) (z0 : Int) (y : Fin 2 → Int) (output : Int)
    (h : W_Ideal.generatedRoot params (z0, (y, ())) output) :
    output = entry table1 (entry table0 (z0 + y 0) + 1 + y 1) := by
  obtain ⟨witness, hbody⟩ := h
  simp only [W_Ideal.generatedRoot.body, W_Ideal.generatedRoot.constraints_0] at hbody
  obtain ⟨⟨p0, hp0, h4⟩, -, -, -, hsel0, ⟨p1, hp1, h23⟩, -, -, -, hsel1, hout⟩ := hbody
  have hp0' : p0 = 0 := by fin_cases p0 <;> simp_all
  have hp1' : p1 = 1 := by fin_cases p1 <;> simp_all
  subst hp0' hp1'
  have h20 := select_entry hsel0
  have h39 := select_entry hsel1
  rw [hout, h39, h20, h23, h4]
  simp [entry, table0, table1]

end Dio2Proof
