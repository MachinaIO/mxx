"""Write `Part_<c_j>_RootSums.lean`: the member sums of the online root (`332`: the mask's placed
members; the fresh columns; every decoder's placed smudge digits), each the loop
`sequential_generatedRoot_K` over members `2, …, w - 1` after members `0` and `1`.

Usage: `python3 gen_root_sums.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from gen_sums import modulus
from patterns import destruct_inline, parse_inline

d = sys.argv[1]
lut_a = '[114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]'
JOBS = [('online_eval', 6, '', 'RootSums'), ('preprocess', 4, 'pre', 'PreRootSums')]
for part in parts_of(d):
  for stage, nf, tag, fname in JOBS:
    path = f'{d}/generated/Stage_{stage}_{part}.lean'
    ns = f'Stage_{stage}_{part}'
    text = open(path).read()
    L = [f'''import TopLemmas

/-! The member sums of the online root (written by `generator/gen_root_sums.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

''']
    found = []
    for k in sorted({int(k) for k in re.findall(r'^def sequential_generatedRoot_(\d+) ', text, re.M)}):
        c0 = text[text.index(f'abbrev sequential_generatedRoot_{k}.constraints_0 '):]
        if lut_a in c0[:c0.index('\n\n')]:
            continue
        body = text[text.index(f'def sequential_generatedRoot_{k} '):]
        fam = re.search(r'let w_1_0 : Fin (\d+) → Mxx\.Primitives\.ExactMatrix (\d+) 8 1 (\d+)', body)
        w, q, cols = int(fam.group(1)), modulus(fam.group(2)), int(fam.group(3))
        M = f'ExactMatrix {q} N 1 {cols}'
        B = f'Blocks {q} {cols}' if nf == 6 else f'Keys {q} {cols}'
        name = f'sequential_generatedRoot_{k}'
        sd = destruct_inline(path, ns, name, 'h')
        _, sn, sa = parse_inline(path, name)
        sgets = [n for n, a in zip(sn, sa) if n != '-' and a.startswith('MxxRuntime.familyGetDynamic')]
        gs = [f'g{j}' for j in range(nf)]
        cps = ['cur.1', 'cur.2.1', 'cur.2.2.1', 'cur.2.2.2.1', 'cur.2.2.2.2.1', 'cur.2.2.2.2.2.1'][:nf]
        L.append(f'''theorem {tag}rootStep{k} {{model : HashModel}} {{params}} {{i : Nat}} {{cur next : {B}}}
    {{{' '.join(gs)} : Fin {w} → {M}}}
    (h : {ns}.{name} Backend.backend model params i
      ({', '.join(cps)}, {', '.join(gs)},
        ()) next) :
    next = cur + ({", ".join(f"cellAt {x} (2 + i)" for x in gs)}, ()) := by
{sd}''')
        for j, n in enumerate(sgets):
            L.append(f'  have e{j} := familyGet_cellAt {n}\n')
        L.append(f'  subst {" ".join(f"e{j}" for j in range(len(sgets)))} hout\n  rfl\n\n')
        hs = ''.join(f'\n    (ha{j} : familyGetStatic g{j} 0 a{j}) (hb{j} : familyGetStatic g{j} 1 b{j})'
                     for j in range(nf))
        ab_ = ' '.join(f'a{j} b{j}' for j in range(nf))
        curs = ['current.1', 'current.2.1', 'current.2.2.1', 'current.2.2.2.1', 'current.2.2.2.2.1',
                'current.2.2.2.2.2.1'][:nf]
        L.append(f'''set_option maxRecDepth 65536 in
/-- The root sum `{k}`: the sums of the families over their {w} members. -/
theorem {tag}rootSum{k} {{model : HashModel}} {{params}} {{{' '.join(gs)} : Fin {w} → {M}}}
    {{{ab_} : {M}}}{hs}
    {{final : {B}}}
    (hruns : MxxIR.IterRuns (fun (i : Nat) (current next : {B}) ↦
      {ns}.{name} Backend.backend model params i
        ({', '.join(curs)}, {', '.join(gs)}, ()) next) (Int.toNat {w - 2})
      ({', '.join(f'matrixAdd a{j} b{j}' for j in range(nf))}, ()) final) :
    final = ({', '.join(f'∑ k, g{j} k' for j in range(nf))}, ()) := by
  have hloop := iter_sum (fun i ↦ ({", ".join(f"cellAt {x} (2 + i)" for x in gs)}, ()))
    (fun i current next hbody ↦ {tag}rootStep{k} hbody) hruns
  rw [show Int.toNat {w - 2} = {w - 2} from rfl] at hloop
''')
        for j in range(nf):
            L.append(f'  obtain ⟨_, ea{j}⟩ := familyGet_eq (k := 0) ha{j}\n')
            L.append(f'  obtain ⟨_, eb{j}⟩ := familyGet_eq (k := 1) hb{j}\n')
        es = ', '.join(f'ea{j}, eb{j}' for j in range(nf))
        L.append(f'''  rw [hloop]
  simp only [{es}, matrixAdd]
  refine {"Prod.ext ?_ (" * nf + "rfl" + ")" * nf} <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
''')
        for j in range(nf):
            L.append(f'  · exact sum_cells (w := {w - 2}) g{j}\n')
        L.append('\n')
        found.append(k)
    L.append(f'end RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_{fname}.lean', 'w').write(''.join(L))
    print(part, stage, found)
  break
