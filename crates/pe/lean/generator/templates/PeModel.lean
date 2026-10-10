import PePhase

/-!
# The model of a run

Every value that precedes key generation, as an explicit function of the hash model, the hash key,
and the sample tape: the anchor `B`, the secret `s`, the attribute ciphertext, the Dynamic packages,
the public keys of the predicate, and the online output `c_F`. Its error `e_F` and the helper error
`e_h` read only the tape keys of `preKeys`, which holds no key of key generation.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The width of TrapGen. -/
abbrev trapSigma : Rat := {{TRAPSIGMA}}

theorem layout0_width : 1 * (Backend.layout0.digitCount + 2) = {{MB}} := by decide

/-- The anchor `B = [ā | I | G - (ā r + e)]` on the tape. -/
noncomputable def BOf (ω : SampleTape) : ExactMatrix Q N 1 {{MB}} :=
  castMatrixColumns layout0_width (trapGenPublic Backend.layout0
    (tapeMatrix ω [0, 0, 2] (.residue Q))
    (tapeMatrix ω [0, 0, 0] (.gaussian trapSigma trapGenCutoff))
    (tapeMatrix ω [0, 0, 1] (.gaussian trapSigma trapGenCutoff)))

/-- The secret on the tape. -/
noncomputable def secOf (ω : SampleTape) : ErrorPoly N :=
  tapeWitness (n := N) (rows := 1) (columns := 1) ω [2, 0] (.interval (-1) 1) 0 0

/-- The gadget row. -/
noncomputable def g0 : ExactMatrix Q N 1 {{D}} :=
  Classical.choose (gadget_exists (m := {{D}}) (base := {{BASE}}) layoutQ rfl (by decide) (by decide))

theorem g0_runs : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g0 :=
  Classical.choose_spec (gadget_exists (m := {{D}}) (base := {{BASE}}) layoutQ rfl (by decide) (by decide))

/-- The public scalar actions `V_1 = Dec(D_1 g)` and `V_K = Dec(D_K g)` of the table `(t₀, t₁)`. -/
noncomputable def V1c (t0 t1 : Int) : ExactMatrix Q N {{D}} {{D}} :=
  Classical.epsilon fun V ↦ gadgetDecomposeRuns Backend.backend {{BASE}} {{D}}
    (matrixMulScalarLeft (lutScalarOne t0 t1) g0) V

noncomputable def VKc (t0 t1 : Int) : ExactMatrix Q N {{D}} {{D}} :=
  Classical.epsilon fun V ↦ gadgetDecomposeRuns Backend.backend {{BASE}} {{D}}
    (matrixMulScalarLeft (lutScalar t0 t1 (substituteMatrix K hK)) g0) V

theorem V1c_eq {t0 t1 : Int} {g : ExactMatrix Q N 1 {{D}}} {V : ExactMatrix Q N {{D}} {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    (hV : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (matrixMulScalarLeft (lutScalarOne t0 t1) g) V) :
    V = V1c t0 t1 := by
  rw [gadgetMatrixRuns_unique hg g0_runs] at hV
  exact gadgetDecomposeRuns_unique hV (Classical.epsilon_spec (p := fun V ↦
    gadgetDecomposeRuns Backend.backend {{BASE}} {{D}} (matrixMulScalarLeft (lutScalarOne t0 t1) g0) V) ⟨V, hV⟩)

theorem VKc_eq {t0 t1 : Int} {g : ExactMatrix Q N 1 {{D}}} {V : ExactMatrix Q N {{D}} {{D}}}
    (hg : gadgetMatrixRuns Backend.backend {{BASE}} {{D}} g)
    (hV : gadgetDecomposeRuns Backend.backend {{BASE}} {{D}}
      (matrixMulScalarLeft (lutScalar t0 t1 (substituteMatrix K hK)) g) V) : V = VKc t0 t1 := by
  rw [gadgetMatrixRuns_unique hg g0_runs] at hV
  exact gadgetDecomposeRuns_unique hV (Classical.epsilon_spec (p := fun V ↦
    gadgetDecomposeRuns Backend.backend {{BASE}} {{D}}
      (matrixMulScalarLeft (lutScalar t0 t1 (substituteMatrix K hK)) g0) V) ⟨V, hV⟩)

/-- The tape samples that precede key generation. -/
structure Prims where
  sec : ErrorPoly N
  B : ExactMatrix Q N 1 {{MB}}
  eb : ErrorMatrix N 1 {{MB}}
  Z : ErrorMatrix N {{MB}} {{M}}
  e0 : Fin 2 → ErrorMatrix N 1 {{D}}
  eh : Fin 2 → ErrorMatrix N 1 {{D}}
  ez : Fin {{M}} → Fin {{M}} → Fin 2 → ErrorMatrix N 1 {{D}}
  eza : Fin {{M}} → Fin {{M}} → Fin 2 → ErrorMatrix N 1 {{D}}

/-- The helper errors. -/
noncomputable def ehOf (ω : SampleTape) : Fin 2 → ErrorMatrix N 1 {{D}} :=
  ![errAt ω [4, 4, 0, 12] helperLaw, errAt ω [4, 4, 0, 20] helperLaw]

/-- The errors of the state encoding. -/
noncomputable def e0Of (ω : SampleTape) : Fin 2 → ErrorMatrix N 1 {{D}} :=
  ![errAt ω [4, 25] encLaw, errAt ω [4, 35] encLaw]

/-- The errors of the digit companions of the attribute ciphertext. -/
noncomputable def ezOf (ω : SampleTape) (jj a : Fin {{M}}) : Fin 2 → ErrorMatrix N 1 {{D}} :=
  ![errAt ω [4, 43, (jj : Nat) + {{M}} * (a : Nat), 26] encLaw,
    errAt ω [4, 43, (jj : Nat) + {{M}} * (a : Nat), 35] encLaw]

/-- The errors of the digit companions of the switching ciphertext. -/
noncomputable def ezaOf (ω : SampleTape) (jj a : Fin {{M}}) : Fin 2 → ErrorMatrix N 1 {{D}} :=
  ![errAt ω [4, 8, (jj : Nat) + {{M}} * (a : Nat), 32] encLaw,
    errAt ω [4, 8, (jj : Nat) + {{M}} * (a : Nat), 41] encLaw]

/-- The samples of a tape. -/
noncomputable def prims (ω : SampleTape) : Prims where
  sec := secOf ω
  B := BOf ω
  eb := ebAt ω [3]
  Z := zAt ω [3]
  e0 := e0Of ω
  eh := ehOf ω
  ez := ezOf ω
  eza := ezaOf ω

section Model

/- The run's attribute `x₀ = xa` and the table `(t₀, t₁)` of the key's program. -/
variable (model : HashModel) (key : ByteArray) (xa : Fin 2) (t0 t1 : Int)

/-- The companion keys `M` of the product. -/
noncomputable def MM : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}} :=
  fun k j _ l ↦ ∑ a, keySum (rhsKey model key) (inputKey model key) k ⟨{{D}} * (j : Nat) + l, by omega⟩ a

/-- The switched keys `T` of the branch of `K`. -/
noncomputable def TM : Fin 2 → ExactMatrix Q N 1 {{M}} :=
  fun k _ p ↦ ∑ a, keySum (autKey model key) (routedKey (MM model key)) k p a

/-- The output key `A_F`. -/
noncomputable def AFM : Fin 2 → Fin 2 → ExactMatrix Q N 1 {{D}} :=
  outputKey model key (MM model key) (TM model key 0) (TM model key 1) (V1c t0 t1) (VKc t0 t1)

variable (P : Prims)

/-- The anchor encoding `b = s B + B² + e_b`. -/
noncomputable def bP : ExactMatrix Q N 1 {{MB}} :=
  reducePoly Q N (P.sec) • P.B + anchorSecond model key + reduceMatrix Q N 1 {{MB}} (P.eb)

noncomputable def ZP : ExactMatrix Q N {{MB}} {{M}} := reduceMatrix Q N {{MB}} {{M}} (P.Z)

/-- The attribute ciphertext `C = P Z + X^{x₀} G`. -/
noncomputable def CtopP : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  ![rowWindow {{D}} 0 (by omega) (P.B * ZP P) + reducePoly Q N (Xe xa) • g0,
    rowWindow {{D}} {{D}} (by omega) (P.B * ZP P)]

noncomputable def CbotP : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  ![rowWindow {{D}} 0 (by omega) ((anchorSecond model key - bP model key P) * ZP P),
    rowWindow {{D}} {{D}} (by omega) ((anchorSecond model key - bP model key P) * ZP P) +
      reducePoly Q N (Xe xa) • g0]

/-- The encoding of the initial state. -/
noncomputable def c0P : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  encRow (P.sec) (inputKey model key)
    (fun j ↦ reducePoly Q N (((1 : Nat) : Int) * 1 * unitPair (P.sec) j) • g0)
    P.e0

/-- The helper of `K`. -/
noncomputable def hrowP : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  encRow (P.sec) (helperKey model key)
    (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution K hK (P.sec)) j) • g0) P.eh

/-- The digit companions of the attribute ciphertext: digit `a` of column `jj`. -/
noncomputable def zP (jj a : Fin {{M}}) : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  encRow (P.sec) (fun k r ↦ rhsKey model key jj a (2 * k + r))
    (fun r ↦ (colDigits (joinRow (CtopP xa P)) (joinRow (CbotP model key xa P)) jj a 0 *
      reducePoly Q N (unitPair (P.sec) r)) • g0)
    (P.ez jj a)

/-- The digit companions of the switching ciphertext: digit `a` of column `jj`. -/
noncomputable def zaP (jj a : Fin {{M}}) : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  encRow (substitution K hK (P.sec)) (fun k r ↦ autKey model key jj a (2 * k + r))
    (fun r ↦ (colDigits (joinRow (switchTop model key)) (joinRow (switchBot model key (hrowP model key P)))
        jj a 0 * reducePoly Q N (unitPair (substitution K hK (P.sec)) r)) • g0)
    (P.eza jj a)

/-- The column and digit of lane `i`. -/
def laneCol (i : Fin {{L}}) : Fin {{M}} := ⟨i % {{M}}, Nat.mod_lt _ (by norm_num)⟩
def laneRow (i : Fin {{L}}) : Fin {{M}} := ⟨i / {{M}}, by omega⟩

/-- The online output `c_F`. -/
noncomputable def cFP : Fin 2 → ExactMatrix Q N 1 {{D}} :=
  let cIn := c0P model key P
  let Ctop := CtopP xa P
  let Cbot := CbotP model key xa P
  let A := inputKey model key
  let Mmul := MM model key
  let z0 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zP model key xa P (laneCol i) (laneRow i) 0
  let z1 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zP model key xa P (laneCol i) (laneRow i) 1
  let za0 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zaP model key P (laneCol i) (laneRow i) 0
  let za1 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zaP model key P (laneCol i) (laneRow i) 1
  let Emul : ExactMatrix Q N 1 {{M}} := fun _ jj ↦ ∑ a, mulCompanionSum A z0 z1 jj a
  let mulRow : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦
    pairMul cIn (actionDigits (joinRow Ctop) (joinRow Cbot) j) + halves Emul j
  let Elut : ExactMatrix Q N 1 {{M}} := fun _ jj ↦ ∑ a, lutCompanionSum Mmul za0 za1 jj a
  let hrow := hrowP model key P
  let top : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
  let bot : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦
    substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
  let σc : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦ substituteMatrix K hK (mulRow j)
  fun j ↦ mulRow j * V1c t0 t1 + (pairMul hrow (actionDigits (TM model key 0) (TM model key 1) j) +
    (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) + halves Elut j)) * VKc t0 t1

/-- The error `e_F` of the online output. -/
noncomputable def eFP : Fin 2 → ErrorMatrix N 1 {{D}} :=
  fun j ↦ liftMatrix (cFP model key xa t0 t1 P j -
    (reducePoly Q N (P.sec) • AFM model key t0 t1 0 j + AFM model key t0 t1 1 j))

end Model

/-! ## The model of a tape -/

section Tape

variable (model : HashModel) (key : ByteArray) (xa : Fin 2) (t0 t1 : Int) (ω : SampleTape)

noncomputable def bOf := bP model key (prims ω)
noncomputable def ZOf := ZP (prims ω)
noncomputable def CtopOf := CtopP xa (prims ω)
noncomputable def CbotOf := CbotP model key xa (prims ω)
noncomputable def c0Of := c0P model key (prims ω)
noncomputable def hrowOf := hrowP model key (prims ω)
noncomputable def zOf := zP model key xa (prims ω)
noncomputable def zaOf := zaP model key (prims ω)
noncomputable def cFOf := cFP model key xa t0 t1 (prims ω)
noncomputable def eFOf := eFP model key xa t0 t1 (prims ω)

end Tape

end PeProof

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

section Unfold

variable (model : HashModel) (key : ByteArray) (xa : Fin 2) (t0 t1 : Int) (ω : SampleTape)

theorem bOf_eq : bOf model key ω =
    reducePoly Q N (secOf ω) • BOf ω + anchorSecond model key + reduceMatrix Q N 1 {{MB}} (ebAt ω [3]) := by
  simp only [bOf, bP, prims]

theorem CtopOf_eq : CtopOf xa ω = ![rowWindow {{D}} 0 (by omega) (BOf ω * reduceMatrix Q N {{MB}} {{M}} (zAt ω [3])) +
      reducePoly Q N (Xe xa) • g0,
    rowWindow {{D}} {{D}} (by omega) (BOf ω * reduceMatrix Q N {{MB}} {{M}} (zAt ω [3]))] := by
  simp only [CtopOf, CtopP, ZP, prims]

theorem CbotOf_eq : CbotOf model key xa ω =
    ![rowWindow {{D}} 0 (by omega) ((anchorSecond model key - bOf model key ω) * reduceMatrix Q N {{MB}} {{M}} (zAt ω [3])),
      rowWindow {{D}} {{D}} (by omega) ((anchorSecond model key - bOf model key ω) *
        reduceMatrix Q N {{MB}} {{M}} (zAt ω [3])) + reducePoly Q N (Xe xa) • g0] := by
  simp only [CbotOf, CbotP, ZP, bOf, prims]

theorem c0Of_eq : c0Of model key ω = encRow (secOf ω) (inputKey model key)
    (fun j ↦ reducePoly Q N (((1 : Nat) : Int) * 1 * unitPair (secOf ω) j) • g0)
    ![errAt ω [4, 25] encLaw, errAt ω [4, 35] encLaw] := by
  simp only [c0Of, c0P, prims, e0Of]

theorem hrowOf_eq : hrowOf model key ω = encRow (secOf ω) (helperKey model key)
    (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      unitPair (substitution K hK (secOf ω)) j) • g0) (ehOf ω) := by
  simp only [hrowOf, hrowP, prims]

theorem zOf_eq (jj a : Fin {{M}}) : zOf model key xa ω jj a =
    encRow (secOf ω) (fun k r ↦ rhsKey model key jj a (2 * k + r))
      (fun r ↦ (colDigits (joinRow (CtopOf xa ω)) (joinRow (CbotOf model key xa ω)) jj a 0 *
        reducePoly Q N (unitPair (secOf ω) r)) • g0)
      ![errAt ω [4, 43, (jj : Nat) + {{M}} * (a : Nat), 26] encLaw,
        errAt ω [4, 43, (jj : Nat) + {{M}} * (a : Nat), 35] encLaw] := by
  simp only [zOf, zP, CtopOf, CbotOf, prims, ezOf]

theorem zaOf_eq (jj a : Fin {{M}}) : zaOf model key ω jj a =
    encRow (substitution K hK (secOf ω)) (fun k r ↦ autKey model key jj a (2 * k + r))
      (fun r ↦ (colDigits (joinRow (switchTop model key)) (joinRow (switchBot model key (hrowOf model key ω)))
          jj a 0 * reducePoly Q N (unitPair (substitution K hK (secOf ω)) r)) • g0)
      ![errAt ω [4, 8, (jj : Nat) + {{M}} * (a : Nat), 32] encLaw,
        errAt ω [4, 8, (jj : Nat) + {{M}} * (a : Nat), 41] encLaw] := by
  simp only [zaOf, zaP, hrowOf, prims, ezaOf]

theorem cFOf_eq : cFOf model key xa t0 t1 ω =
    let cIn := c0Of model key ω
    let Ctop := CtopOf xa ω
    let Cbot := CbotOf model key xa ω
    let A := inputKey model key
    let Mmul := MM model key
    let z0 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zOf model key xa ω (laneCol i) (laneRow i) 0
    let z1 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zOf model key xa ω (laneCol i) (laneRow i) 1
    let za0 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zaOf model key ω (laneCol i) (laneRow i) 0
    let za1 : Fin {{L}} → ExactMatrix Q N 1 {{D}} := fun i ↦ zaOf model key ω (laneCol i) (laneRow i) 1
    let Emul : ExactMatrix Q N 1 {{M}} := fun _ jj ↦ ∑ a, mulCompanionSum A z0 z1 jj a
    let mulRow : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦
      pairMul cIn (actionDigits (joinRow Ctop) (joinRow Cbot) j) + halves Emul j
    let Elut : ExactMatrix Q N 1 {{M}} := fun _ jj ↦ ∑ a, lutCompanionSum Mmul za0 za1 jj a
    let hrow := hrowOf model key ω
    let top : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦ substituteMatrix K hK (helperKey model key 0 j)
    let bot : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦
      substituteMatrix K hK (helperKey model key 1 j) - substituteMatrix K hK (hrow j)
    let σc : Fin 2 → ExactMatrix Q N 1 {{D}} := fun j ↦ substituteMatrix K hK (mulRow j)
    fun j ↦ mulRow j * V1c t0 t1 + (pairMul hrow (actionDigits (TM model key 0) (TM model key 1) j) +
      (pairMul σc (actionDigits (joinRow top) (joinRow bot) j) + halves Elut j)) * VKc t0 t1 := by
  simp only [cFOf, cFP, c0Of, CtopOf, CbotOf, zOf, zaOf, hrowOf]

theorem eFOf_eq (j : Fin 2) : eFOf model key xa t0 t1 ω j = liftMatrix (cFOf model key xa t0 t1 ω j -
    (reducePoly Q N (secOf ω) • AFM model key t0 t1 0 j + AFM model key t0 t1 1 j)) := by
  simp only [eFOf, eFP, cFOf, prims]

end Unfold

end PeProof
