"""Write `Part_<c_j>_Sums.lean` for every part of parameter directory `d`: for every member sum of
the online evaluation (a root parallel `parallel_generatedRoot_K` over the members that gathers
member `o`'s cells with `parallel_parallel_generatedRoot_K_G` and adds them with the loop
`sequential_parallel_generatedRoot_K_L`), the gather, one loop step, and the sum.

Usage: `python3 gen_sums.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import destruct, destruct_inline, parse, parse_inline

d = sys.argv[1]
from gen_part import parts_of
parts = parts_of(d)
abbrevs = {}
setup = open(f'{d}/TopSetup.lean').read()
for name, value in re.findall(r'abbrev (\w+) : Nat := (\d+)', setup):
    abbrevs[value] = name


def modulus(q):
    return abbrevs.get(q, q)


def sums(path):
    """The member sums of the online root: `(K, G, L, members, width, q, columns)`."""
    text = open(path).read()
    out = []
    for k in sorted(set(re.findall(r'^structure parallel_generatedRoot_(\d+)\.Witness', text, re.M)), key=int):
        name = f'parallel_generatedRoot_{k}'
        body = text[text.index(f'abbrev {name}.body'):]
        body = body[:body.index('\n\n')]
        g = re.search(rf'parallel_parallel_generatedRoot_{k}_(\d+) backend', text[text.index(f'abbrev {name}.constraints_0'):])
        l = re.search(rf'sequential_parallel_generatedRoot_{k}_(\d+) backend', text[text.index(f'abbrev {name}.constraints_0'):])
        if not g or not l:
            continue
        fam = re.search(r'let w_0_0 : Fin (\d+) → Mxx\.Primitives\.ExactMatrix (\w+) 8 1 (\d+) := inputs\.1', body)
        if not fam:
            continue
        cells, q, cols = int(fam.group(1)), fam.group(2), int(fam.group(3))
        gname = f'parallel_parallel_generatedRoot_{k}_{g.group(1)}'
        gtext = text[text.index(f'def {gname} '):]
        width = int(re.search(r'\(\((\d+) \* \(Int\.ofNat i_0\)\) \+ \(Int\.ofNat i_1\)\)', gtext).group(1))
        out.append((k, g.group(1), l.group(1), cells // width, width, q, cols))
    return out


def lemmas(path, ns, k, g, l, members, width, q, cols, nf=6, tag=''):
    M = f'ExactMatrix {modulus(q)} N 1 {cols}'
    B = f'Blocks {modulus(q)} {cols}' if nf == 6 else f'Keys {modulus(q)} {cols}'
    cells = members * width
    gd = destruct_inline(path, ns, f'parallel_parallel_generatedRoot_{k}_{g}', 'h')
    sd = destruct_inline(path, ns, f'sequential_parallel_generatedRoot_{k}_{l}', 'h')
    md = destruct(path, ns, f'parallel_generatedRoot_{k}', 'h')
    _, gn, ga = parse_inline(path, f'parallel_parallel_generatedRoot_{k}_{g}')
    gets = [n for n, a in zip(gn, ga) if n != '-' and a.startswith('MxxRuntime.familyGetDynamic')]
    _, sn, sa = parse_inline(path, f'sequential_parallel_generatedRoot_{k}_{l}')
    sgets = [n for n, a in zip(sn, sa) if n != '-' and a.startswith('MxxRuntime.familyGetDynamic')]
    _, mn, ma = parse(path, f'parallel_generatedRoot_{k}')
    gather = next(n for n, a in zip(mn, ma) if a.startswith('(∀ i'))
    loop = next(n for n, a in zip(mn, ma) if a.startswith('MxxIR.IterRuns'))
    statics = [(n, a) for n, a in zip(mn, ma) if a.startswith('MxxRuntime.familyGetStatic')]
    text = open(path).read()
    body = text[text.index(f'abbrev parallel_generatedRoot_{k}.body'):]
    body = body[:body.index('\n\n')]
    fams = re.findall(r'let (w_\d+_\d+) := witness\.(w_\d+_\d+)', body)
    famnames = [a for a, _ in fams if re.match(r'w_\d+_[0-5]$', a)][:nf]
    inits = re.findall(r'let (w_\d+_0) : [^\n]*:= MxxRuntime\.matrixAdd (w_\d+_0) (w_\d+_0)', body)
    count = width - 2
    idx = '(cellIndex (⟨o, ho⟩ : Fin {m}) (⟨i, hi⟩ : Fin {w}))'.format(m=members, w=width)
    fs = [f'f{j}' for j in range(nf)]
    L = []
    fl = ' '.join(fs)
    ft = ', '.join(fs)
    L.append(f'''theorem {tag}gather{k} {{model : HashModel}} {{params}} {{o i : Nat}}
    {{{fl} : Fin {cells} → {M}}} {{outputs : {B}}}
    (h : {ns}.parallel_parallel_generatedRoot_{k}_{g} Backend.backend model params o i
      ({ft}, ()) outputs) (ho : o < {members}) (hi : i < {width}) :
    outputs = ({", ".join(f + " " + idx for f in fs)}, ()) := by
{gd}  have hidx : w_1_0 = Int.ofNat ({width} * o + i) := by
    simp only [w_1_0, Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at {" ".join(gets)}
''')
    for n in gets:
        L.append(f'  obtain ⟨_, e{n[1:]}⟩ := familyGet_eq {n}\n')
    L.append(f'  subst {" ".join("e" + n[1:] for n in gets)} hout\n  rfl\n\n')
    gs = [f'g{j}' for j in range(nf)]
    projs_ = ['cur.1', 'cur.2.1', 'cur.2.2.1', 'cur.2.2.2.1', 'cur.2.2.2.2.1', 'cur.2.2.2.2.2.1'][:nf]
    L.append(f'''theorem {tag}step{k} {{model : HashModel}} {{params}} {{o i : Nat}} {{cur next : {B}}}
    {{{' '.join(gs)} : Fin {width} → {M}}}
    (h : {ns}.sequential_parallel_generatedRoot_{k}_{l} Backend.backend model params o i
      ({', '.join(projs_)}, {', '.join(gs)},
        ()) next) :
    next = cur + ({", ".join(f"cellAt {x} (2 + i)" for x in gs)}, ()) := by
{sd}''')
    for j, n in enumerate(sgets):
        L.append(f'  have e{j} := familyGet_cellAt {n}\n')
    L.append(f'  subst {" ".join(f"e{j}" for j in range(len(sgets)))} hout\n  rfl\n\n')
    sums_ = ', '.join(f'∑ k : Fin {width}, {f} (cellIndex o k)' for f in fs)
    L.append(f'''set_option maxRecDepth 65536 in
/-- Member `o` of the sum `{k}`: the sum of its {width} cells. -/
theorem {tag}members{k} {{model : HashModel}} {{params}} {{o : Fin {members}}}
    {{{fl} : Fin {cells} → {M}}} {{outputs : {B}}}
    (h : {ns}.parallel_generatedRoot_{k} Backend.backend model params o
      ({ft}, ()) outputs) :
    outputs = ({sums_}, ()) := by
{md}''')
    w6 = famnames
    conj = ' ∧\n      '.join(f'{w6[j]} i = {fs[j]} (cellIndex o i)' for j in range(nf))
    L.append(f'''  have hg : ∀ i : Fin {width}, {conj} := by
    intro i
    have := {tag}gather{k} ({gather} i) o.isLt i.isLt
    simp only [Prod.mk.injEq] at this
    obtain ⟨{', '.join(f'a{j}' for j in range(nf))}, -⟩ := this
    exact ⟨{', '.join(f'a{j}' for j in range(nf))}⟩
''')
    projs = (['.1', '.2.1', '.2.2.1', '.2.2.2.1', '.2.2.2.2.1', '.2.2.2.2.2'] if nf == 6 else
             ['.1', '.2.1', '.2.2.1', '.2.2.2'])
    for j in range(nf):
        L.append(f'  have hw{j} : {w6[j]} = fun i ↦ {fs[j]} (cellIndex o i) := funext fun i ↦ (hg i){projs[j]}\n')
    es = []
    for n, a in statics:
        kk = re.search(r'familyGetStatic \S+ \((\d+)\)', a).group(1)
        L.append(f'  obtain ⟨_, e{n[1:]}⟩ := familyGet_eq (k := {kk}) {n}\n')
        es.append('e' + n[1:])
    L.append(f'''  have hloop := iter_sum (fun i ↦ ({", ".join(f"cellAt {x} (2 + i)" for x in w6)}, ()))
    (fun i current next hbody ↦ {tag}step{k} hbody) {loop}
  rw [show Int.toNat {count} = {count} from rfl] at hloop
  rw [hout, hloop]
  simp only [{", ".join(a for a, _, _ in inits)}, {", ".join(es)}, {", ".join(f"hw{j}" for j in range(nf))}]
  simp only [matrixAdd]
  refine {"Prod.ext ?_ (" * nf + "rfl" + ")" * nf} <;>
    simp only [Prod.fst_add, Prod.snd_add, Prod.fst_sum, Prod.snd_sum, cellAt]
''')
    for f in fs:
        L.append(f'  · exact sum_cells (w := {count}) (fun k ↦ {f} (cellIndex o k))\n')
    L.append('\n')
    return ''.join(L)


if __name__ == '__main__':
  for part in parts:
      path = f'{d}/generated/Stage_online_eval_{part}.lean'
      ns = f'Stage_online_eval_{part}'
      found = sums(path)
      out = [f'''import TopLemmas

/-!
# The member sums of the online evaluation (written by `generator/gen_sums.py`)

Every PRF bucket sums the cells of each member: member `o`'s cells `w o, …, w o + w - 1` of the
bucket's cell family are gathered, the first two added, and the rest added by a loop.
-/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

''']
      for args in found:
          out.append(lemmas(path, ns, *args))
      out.append(f'end RefreshTopProof.Part_{part}\n')
      open(f'{d}/Part_{part}_Sums.lean', 'w').write(''.join(out))
      print(part, [a[0] for a in found])
