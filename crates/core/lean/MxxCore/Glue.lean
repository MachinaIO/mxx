import MathcalA.Lib.Sym

/-!
# Program-logic rules for the glue of proof images

mathcal-A's `Prog.Spec` rules cover straight-line code, branches, and `JAL`. The glue of a proof
image also calls leaves and its copy routine with `auipc ra, hi` and `jalr ra, lo(ra)`, and
they return with `jalr x0, 0(ra)`; these are the rules for those two instructions.
-/

namespace MxxCore

open MathcalA RiscvZkvm.Rv64 Machine

namespace Glue

variable {P : Prog}

/-- `AUIPC rd imm`. -/
theorem auipc {j k : Nat} {s : MachineState} {n : Nat} {Q : MachineState → Prop} {rd : Reg}
    {imm : BitVec 20} (hi : P.instrs[j]? = some (.AUIPC rd imm))
    (h : P.Spec (j + 1)
      (s.setReg rd (pcOf j + ((imm.zeroExtend 32 <<< 12 : BitVec 32).signExtend 64))) n k Q) :
    P.Spec j s (n + 1) k Q := by
  refine Prog.Spec.next ?_ h
  simp only [Machine.step, Prog.fetch hi, forbidden, Bool.false_eq_true, ↓reduceIte,
    Machine.memAccess, reduceCtorEq, beq_iff_eq]
  congr 1
  simp only [execInstrBr, Prog.mach, Prog.pcOf_succ, Prog.setReg_with, MachineState.setPC]

/-- `JALR rd rs1 off`, to instruction `t`. -/
theorem jalr {j k t : Nat} {s : MachineState} {n : Nat} {Q : MachineState → Prop} {rd rs1 : Reg}
    {off : BitVec 12} (hi : P.instrs[j]? = some (.JALR rd rs1 off))
    (ht : (s.getReg rs1 + signExtend12 off) &&& ~~~1#64 = pcOf t)
    (h : P.Spec t (s.setReg rd (pcOf (j + 1))) n k Q) : P.Spec j s (n + 1) k Q := by
  refine Prog.Spec.next ?_ h
  simp only [Machine.step, Prog.fetch hi, forbidden, Bool.false_eq_true, ↓reduceIte,
    Machine.memAccess, reduceCtorEq, beq_iff_eq]
  congr 1
  simp only [execInstrBr, Prog.mach, Prog.pcOf_succ, Prog.setReg_with, MachineState.setPC,
    Prog.getReg_with, ht]

end Glue

end MxxCore
