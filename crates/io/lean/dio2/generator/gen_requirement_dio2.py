#!/usr/bin/env python3
"""Write `proof/Dio2Requirement.lean`: the requirement facts of the dio2 claim.

Runs refresh_top's `gen_requirement2.py` on the worst-case module `W_Requirement_0` of an
average-case package (namespace `Dio2Requirement`). The root conditions of the dio2 requirement are
conjunctions the generator states unsplit, as sums of `if … then 1 else 0`; their numerals are
typed `Int`, as in the export.

The guarded dio2 overlay also handles parallel integer cells and deterministic
SHA column families in v2 exports, without modifying the shared generator.

Usage: gen_requirement_dio2.py <average package>
"""
import os
import re
import sys
from pathlib import Path

import xroles_dio2
import dio2_meta

here = os.path.dirname(os.path.abspath(__file__))
pkg = sys.argv[1]
metadata = dio2_meta.extract(Path(pkg))
if 'worst/W_Requirement_0.lean' not in metadata['sources']:
    raise RuntimeError('Metadata has no worst-case requirement')
out = os.path.join(pkg, 'proof', 'Dio2Requirement.lean')
Path(out).parent.mkdir(parents=True, exist_ok=True)
source = Path(xroles_dio2.REFRESH_TOP, 'gen_requirement2.py')
code = source.read_text()


def extend(old, new):
    global code
    if code.count(old) != 1:
        raise RuntimeError('Unsupported refresh_top requirement generator: ' + old[:80])
    code = code.replace(old, new)


extend("scopes = xmodel.parse_module(gdir, MOD)", """scopes = xmodel.parse_module(gdir, MOD)
parents = {atom.loop.child: atom.loop.kind
           for scope in scopes.values() for atom in scope.loops}
""")
extend("    if is_root:\n        I.cvars", "    if is_root or parents.get(sc.name) == 'par':\n        I.cvars")
extend("        elif h == 'MxxRuntime.select':\n            I.defn", """        elif h == 'MxxRuntime.sha256BitColumns':
            I.defn[strip_parens(at.args[4])] = ('sha0', j)
            I.defn[strip_parens(at.args[5])] = ('sha1', j)
        elif h == 'MxxRuntime.select':
            I.defn""")
extend("    lp = at.loop\n    ch = INFO[lp.child]", """    if kind in ('sha0', 'sha1'):
        arguments = ' '.join(paren(expr(I, argument, deep, seen)) for argument in at.args[:4])
        columns = paren(expr(I, at.args[2], deep, seen))
        projection = 1 if kind == 'sha0' else 2
        return f'(fun index : Fin {columns} ↦ ((Sha256Columns.column {arguments} index.val).{projection} : Int))'
    lp = at.loop
    ch = INFO[lp.child]""")
extend("            check(at.loop.kind == 'seq', f'{sc.name}: loop kind {at.loop.kind}')\n            fin = strip_parens(at.loop.final)", """            if at.loop.kind == 'par':
                fin = WIRE.search(at.loop.outs[0]).group(0)
                I.defn[fin] = ('parallel', j)
                continue
            check(at.loop.kind == 'seq', f'{sc.name}: loop kind {at.loop.kind}')
            fin = strip_parens(at.loop.final)""")
extend("    init = expr(I, strip_parens(lp.init), deep, seen)", """    if kind == 'parallel':
        term = subst(ch.T, child_mapping(I, at, ch, deep, seen))
        return f'(fun member : Fin {lp.count} ↦ {term})'
    init = expr(I, strip_parens(lp.init), deep, seen)""")
extend("    m = re.search(r'=> (.*) next\\) \\(Int\\.toNat', at.text, re.S)", """    if at.loop.kind == 'par':
        body = strip_parens(at.text).split(', ', 1)[1]
        head, args = app_split(body)
        mapping = {}
        for variable, argument in zip(ch.bvars, args[1:-2]):
            argument = strip_parens(argument)
            mapping[variable] = 'member.val' if argument == at.loop.var else expr(I, argument, deep, seen)
        for variable, argument in zip(ch.cvars, tuple_items(args[-2])):
            mapping[variable] = expr(I, strip_parens(argument), deep, seen)
        return mapping
    m = re.search(r'=> (.*) next\\) \\(Int\\.toNat', at.text, re.S)""")
extend("        elif kind == 'select':\n            pf = f'select_int {names[j]}'", """        elif kind == 'select':
            pf = f'select_int {names[j]}'
        elif kind in ('sha0', 'sha1'):
            projection = 1 if kind == 'sha0' else 2
            pf = f'by funext index; exact ({names[j]}.2.2.2.2.2 index).{projection}'
        elif kind == 'parallel':
            ch = INFO[at.loop.child]
            pf = f'by funext member; exact value_{ch.sc.name} ({names[j]} member)'""")
extend("    if not I.root:\n        bs.append", "    if not I.root and I.state is not None:\n        bs.append")
extend("links.append(('and', strip_parens(expr(I, mc.group(1), True))))",
       "links.append(('and', strip_parens(expr(I, mc.group(1), True)).replace('then 1 else 0', 'then (1 : Int) else 0')))")
extend("    for v, t in zip(I.cvars, I.sc.inputs):\n        if v != 'cur':", """    if not I.root and I.state is None:
        bs.append('{next : Int}')
    for v, t in zip(I.cvars, I.sc.inputs):
        if v != 'cur':""")
extend("    if I.state == 'Int':\n        kind, t", """    if I.state is None:
        I.T = value(I, outs[0], True)
        body = dl + equations(I, names) + ['  exact hout']
        blocks.append(f'''/-- The integer output of `{n}`, without unrolling its sums. -/
theorem value_{n} {binders_text(I)}
    (h : {call(I, 'next')}) : next = {I.T} := by
''' + '\\n'.join(body))
        continue
    if I.state == 'Int':
        kind, t""")
sys.argv = [str(source), pkg, '--gdir', os.path.join(pkg, 'worst'), '--module', 'W_Requirement_0',
            '--namespace', 'Dio2Requirement', '--out', out]
exec(compile(code, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
text = open(out).read()
text = re.sub(r'^(  cond_\d+ : .*)$', lambda m: m.group(1).replace('then 1 else 0)', 'then (1 : Int) else 0)'),
              text, flags=re.M)
open(out, 'w').write(text)
