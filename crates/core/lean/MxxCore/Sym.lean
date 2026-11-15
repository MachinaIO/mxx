import MathcalA.Lib.Sym

/-!
# Symbolic execution of code at any offset

mathcal-A's `mrun` looks each instruction up by evaluating `P.instrs`, so it runs only on a
concrete program from known instruction indices. `mrunAt` runs code that a hypothesis places at
an offset `c` of any program: instruction lookups come from the code hypotheses
(`P.instrs[c + i]? = some …`), and branch and jump targets are proved by word arithmetic under a
hypothesis that bounds `BASE + 4 c` below `2^64`. A piece of code, such as the copy routine of
proof images or a leaf, is then specified once for every image that contains it.
-/

open MathcalA RiscvZkvm.Rv64 Machine Lean.Parser.Tactic

namespace MxxCore

/-- The instruction lookup of a step at an offset: by evaluation, or from a code hypothesis. -/
syntax "mlookup" : tactic
macro_rules
  | `(tactic| mlookup) => `(tactic| first | rfl | assumption |
      (simp only [Nat.add_assoc, Nat.reduceAdd] at *; assumption))

/-- A branch or jump target at an offset: `pcOf j + off = pcOf (target j w off)`, by evaluation or
by word arithmetic, using the hypotheses that bound the offset. -/
syntax "mtarget" : tactic
macro_rules
  | `(tactic| mtarget) => `(tactic| first | decide |
      (simp only [MathcalA.pcOf, MathcalA.target, BitVec.toNat_ofNat, RiscvZkvm.Rv64.reduceSignExtend13,
          RiscvZkvm.Rv64.reduceSignExtend21, Nat.reduceMod, Nat.reducePow, Nat.reduceSub,
          Nat.reduceDiv, Nat.reduceLT, ↓reduceIte, MathcalA.BASE] at *
       rw [← BitVec.ofNat_add]
       exact MathcalA.W_mod (by omega)))

/-- One symbolic step of a `Spec` goal whose code is given by hypotheses. Each rule's lookup and
target are separate goals proved by tactics, so a rule that does not apply fails and the next
is tried. -/
syntax "mstepAt" (" [" (simpStar <|> simpErase <|> simpLemma),* "]")? : tactic
macro_rules
  | `(tactic| mstepAt) => `(tactic| mstepAt [])
  | `(tactic| mstepAt [$ts,*]) => `(tactic|
      first
      | (refine MathcalA.Prog.Spec.step (i := ?i) ?hi ?hseq ?_ ?_
         case' hi => mlookup
         case' hseq => rfl
         on_goal 2 => first
           | (simp only [MathcalA.Prog.MemOk, Machine.memAccess]; done)
           | (try symsimp [MathcalA.Prog.MemOk, Machine.memAccess, $ts,*])
         on_goal 1 => try symsimp [$ts,*])
      | (refine MathcalA.Prog.Spec.branch (i := ?i) (o := ?o) ?hi ?hoff ?ht ?_ ?_
         case' hi => mlookup
         case' hoff => rfl
         case' ht => mtarget
         all_goals
           intro $(Lean.mkIdent `hbr):ident
           try simp only [MathcalA.brCond] at $(Lean.mkIdent `hbr):ident
           try symsimp [$ts,*] at $(Lean.mkIdent `hbr):ident
           try symsimp [$ts,*])
      | (refine MathcalA.Prog.Spec.jal (rd := ?rd) (o := ?o) ?hi ?ht ?_
         case' hi => mlookup
         case' ht => mtarget
         try symsimp [$ts,*]))

/-- `mrun` for code given by hypotheses, at any offset. -/
syntax "mrunAt" (" [" (simpStar <|> simpErase <|> simpLemma),* "]")? : tactic
macro_rules
  | `(tactic| mrunAt) => `(tactic| mrunAt [])
  | `(tactic| mrunAt [$ts,*]) => `(tactic|
      (mstepAt [$ts,*]
       repeat' (first | (refine MathcalA.Prog.Spec.done ?_; try symsimp [$ts,*]) | mstepAt [$ts,*])))

end MxxCore
