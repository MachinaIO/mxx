#!/usr/bin/env python3
"""Generate the two date steps and their composition."""
import re
from date_parameters import reference_facts, write_chain

facts = reference_facts('pre')
start = facts.index("structure generatedRoot.Facts_0 ")
chunk_of = {}
cur = None
for line in facts[start:].splitlines():
    m = re.match(r"structure generatedRoot\.Facts_(\d+) ", line)
    if m:
        cur = int(m.group(1))
        continue
    if line.startswith("structure generatedRoot.Facts "):
        break
    m = re.match(r"  (h[0-9_]+) :", line)
    if m and cur is not None:
        chunk_of.setdefault(m.group(1), cur)

def online(base: int) -> dict:
    """The online root wires of a step whose state concatenation is `base` (not step 0)."""
    return dict(oTI0=str(base + 27), oTI1=str(base + 28), oTL0=str(base + 45),
                oTL1=str(base + 46), oOut0=str(base + 112), oOut1=str(base + 118))


def pre(s: int) -> dict:
    """The preprocessing root wires of step `s`: lane families and the output key."""
    offset = 138 * s
    return dict(pI=str(44 + offset), pL=str(45 + offset), pS0=str(126 + offset),
                pS1=str(132 + offset), pS2=str(138 + offset), pS3=str(144 + offset))


STEPS = {
    0: dict(oR0="8_0", oR1="9_0", oTI0="47", oTI1="48", oTL0="67", oTL1="68",
            oOut0="134", oOut1="140", off="0", pA0="", pA1="", C_="", kind="FIRST", **pre(0)),
    1: dict(oR0="134_0", oR1="140_0", off="128", pA0="147", pA1="148", C_="C", kind="MID", **online(141), **pre(1)),
}
for k in (1,):
    prev = pre(k - 1)
    STEPS[k].update(pP0=prev["pS0"], pP1=prev["pS1"], pP2=prev["pS2"], pP3=prev["pS3"])
STEPS[0].update(pP0="", pP1="", pP2="", pP3="")

TEMPLATE = open("PeChain.lean.in").read()
for k, d in STEPS.items():
    d = dict(d, k=str(k), table="tableConst" if d["C_"] else "tableMono")
    keep = {d["kind"], "CONST" if d["C_"] else "MONO"}
    lines = []
    for line in TEMPLATE.splitlines():
        m = re.search(r" --(FIRST|MID|MONO|CONST)$", line)
        if m:
            if m.group(1) not in keep:
                continue
            line = line[: m.start()]
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text = re.sub(r"«(\w+)»", lambda m: d[m.group(1)], text)
    text = re.sub(r"@@(h[0-9_]+)@@", lambda m: f"G.c{chunk_of[m.group(1)]}.{m.group(1)}", text)
    write_chain(f"PeChain{k}.lean", text)

all_fields = {f"{field}_{step}": value for step, fields in STEPS.items()
              for field, value in fields.items()}
template = open('PeChainAll.lean.in').read()
template = '\n'.join(line for line in template.splitlines()
                     if not re.match(r'(import PeChain[23]|    \(h(?:M2|A[23]|T[23]))', line)) + '\n'
template = re.sub(r'\bFin 4\b', 'Fin 2', template)
template = re.sub(r'\bFin 5\b', 'Fin 3', template)
template = template.replace('v 4', 'v 2').replace('< 384', '< 256')
template = template.replace('_3»', '_1»').replace('four online steps', 'two online steps')
template = template.replace('# The four steps', '# The two date steps')
template = template[:template.index('  obtain ⟨p0,')] + '''  obtain ⟨p0, e0, hp0, he0, hv1⟩ := hchain 0
  obtain ⟨p1, e1, hp1, he1, hv2⟩ := hchain 1
  have c0 : (0 : Fin 2).castSucc = 0 := rfl
  have c1 : (1 : Fin 2).castSucc = 1 := rfl
  have s0 : (0 : Fin 2).succ = 1 := rfl
  have s1 : (1 : Fin 2).succ = 2 := rfl
  rw [c0] at he0
  rw [c1] at he1
  rw [s0] at hv1
  rw [s1] at hv2
  have n0 : ((0 : Fin 2) : Nat) = 0 := rfl
  have n1 : ((1 : Fin 2) : Nat) = 1 := rfl
  rw [n0] at he0
  rw [n1] at he1
  have x0 := hx p0
  have x1 := hx p1
  have hv1b : 0 ≤ v 1 ∧ v 1 < 8 := by
    rw [hv1]; exact ⟨(ht e0).1, ht8 e0 (by omega)⟩
  have he_0 : 0 + (x p0).toNat < 128 := by omega
  have R0 := chain_step0 F G hkey htab hpos hM0 hT0I0 hT0I1 hT0L0 hT0L1 hT0M0 hT0M1 hs henc p0 hp0 (hrhs p0) (heB p0) hcomp0 hhelp
    hcomp (fun j ↦ htN 0 j (by omega)) he_0
  rw [tableMono_eq t 0 _ _ e0 (by simp only; omega), target_mono, ← hv1, ← hA1] at R0
  have he_1 : (v 1).toNat + (x p1).toNat < 128 := by omega
  have R1 := chain_step1 F G hkey htab hpos hA1 hM1 hT1I0 hT1I1 hT1L0 hT1L1 hT1M0 hT1M1 hs R0 p1 hp1 (hrhs p1) (heB p1) hcomp0 hhelp
    hcomp he_1
  rw [tableConst_eq t 128 _ _ e1 (by simp only; omega), target_zero s (by rw [← hv2, hv4])] at R1
  exact R1

end PeProof
'''
text = re.sub(r"«(\w+)»", lambda match: all_fields[match.group(1)],
              template)
write_chain("PeChainAll.lean", text)
