#!/usr/bin/env python3
"""Generate `Gen_Stage_Links.lean`: wire identities between the stages' instance bodies, by data
flow through the Claim's stage wiring (`GeneratedClaim.Runs`).

For each stage `M` whose root runs `∀ i : Fin S, I … i (args i) (outs i)` over an instance body `I`:
* `M.claimArgs external execution` — the Claim's input tuple of `M` (verbatim);
* `M.rootArgs rM i` — the instance input tuple at member `i` (the root loop's arguments);
* `M.rootOuts rM i` — the instance output tuple at member `i` (the root loop's outputs);
* `src_<M>_<w>` for every instance input wire `w`:
    `I.w cM = <source>` with `<source>` the Claim argument feeding it — an external input
    `external.input_k` or a stage output `execution.«stage_k».<proj>` — applied to the member if the
    root passes it per member. Hypotheses: `hin : cM.inputs = M.rootArgs rM i` and
    `hroot : rM.inputs = M.claimArgs external execution` (both `rfl` for contexts built from `Runs`:
    the root context `⟨args, witness⟩` and the instance context of `generatedRoot.h<K>_at`).
* `out_<M>_<w>` for every instance output wire `w`: `I.w cM = <execution.«stage_k».proj> i`.
  Hypotheses: `hout : execution.«stage_k» = M.generatedRoot.outs rM` (the root's `Facts.hout` with
  `Runs`' output) and `hinst : M.rootOuts rM i = I.outs cM` (the instance's `Facts.hout`).
* `link_<A>_<a>_<B>_<b>`: `A.a cA = B.b cB` for every pair of wires of different stages with the
  same source (stage output → input, or two inputs reading the same Claim value) — `src`/`out` chained.
* `ext_<B>_<b>`: when the stage-output source `a` of `B.b` is a pass-through of `A` (an instance input
  wire of `A` returned as output), `B.b cB = <A.a's Claim source>` (`link` then `src`).
Recognition only by the Claim / root / instance structure (no wire ids).

Usage: gen_stage_links.py <generated dir> <out file> [--only-links]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402


def tree(text):
    """A Lean tuple term as a binary tree: ('pair', left, right) / ('unit',) / ('leaf', text)."""
    t = xmodel.strip_parens(text.strip()) if text.strip() != '()' else '()'
    if t == '()':
        return ('unit',)
    if text.strip().startswith('(') and xmodel.matching(text.strip(), 0) == len(text.strip()) - 1:
        inner = text.strip()[1:-1]
        items = [x.strip() for x in xmodel.split_top(inner, ',')]
        if len(items) >= 2:
            right = items[1] if len(items) == 2 else '(' + ', '.join(items[1:]) + ')'
            return ('pair', tree(items[0]), tree(right))
        return tree(inner)
    return ('leaf', text.strip())


def leaves(t, path=''):
    """(path, leaf text) of every leaf; paths are `.1` / `.2` strings."""
    if t[0] == 'pair':
        return leaves(t[1], path + '.1') + leaves(t[2], path + '.2')
    if t[0] == 'leaf':
        return [(path, t[1])]
    return []


def at(t, path):
    for step in re.findall(r'\.([12])', path):
        if t[0] != 'pair':
            return None
        t = t[1] if step == '1' else t[2]
    return t


def term_path(term, head):
    """`head.2.1.2.1` -> `.2.1.2.1` (or None)."""
    m = re.match(r'^' + re.escape(head) + r'((?:\.[12])*)$', term.strip())
    return m.group(1) if m else None


class Stage:
    def __init__(self, gdir, call):
        self.M = call['module']
        self.call = call
        self.ok = False
        try:
            sc = xmodel.load(gdir, self.M)
        except FileNotFoundError:
            return
        root = sc.get('generatedRoot')
        if root is None:
            return
        rj = root.to_json()
        loops = [a for a in rj['atoms'] if a.get('loop') and a['loop']['kind'] == 'par']
        if not loops:
            return
        self.L = loops[0]['loop']
        self.inst = self.L['child']
        self.S = self.L['count']
        self.rw = rj['wires']
        ij = sc[self.inst].to_json()
        self.iw = ij['wires']
        text = open(f'{gdir}/{self.M}_Facts.lean').read()
        m = re.search(r'^def generatedRoot\.outs .*\n  (.*)$', text, re.M)
        self.rootouts = tree(xmodel.norm_wires(m.group(1))) if m else None
        m = re.search(r'^def ' + re.escape(self.inst) + r'\.outs .*\n  (.*)$', text, re.M)
        self.instouts = tree(xmodel.norm_wires(m.group(1))) if m else None
        self.args_tree = tree(group(self.L['args']))
        self.outs_tree = tree(group(self.L['outs']))
        self.claim_tree = tree(group(call['args']))
        self.stage_out = call['output']
        self.params = call['params'][-1]
        self.ok = bool(self.rootouts and self.instouts)

    def sources(self):
        """instance input wire -> (instance path, root path, claim term, indexed?)."""
        out = {}
        for w, v in self.iw.items():
            if v['src'] != 'input':
                continue
            pi = term_path(v['term'], 'inputs')
            leaf = at(self.args_tree, pi) if pi is not None else None
            if not leaf or leaf[0] != 'leaf':
                continue
            parts = leaf[1].split()
            rw = xsem_wire(parts[0])
            if not rw or self.rw.get(rw, {}).get('src') != 'input':
                continue
            pr = term_path(self.rw[rw]['term'], 'inputs')
            cl = at(self.claim_tree, pr) if pr is not None else None
            if not cl or cl[0] != 'leaf':
                continue
            out[w] = (pi, pr, cl[1].strip(), len(parts) > 1)
        return out

    def outputs(self):
        """instance output wire -> (instance-outs path, stage output path, indexed?)."""
        out = {}
        rootleaf = {xsem_wire(t): p for p, t in leaves(self.rootouts)}
        for q, t in leaves(self.outs_tree):
            parts = t.split()
            rw = xsem_wire(parts[0])
            if rw not in rootleaf:
                continue
            il = at(self.instouts, q)
            if not il or il[0] != 'leaf':
                continue
            iw = xsem_wire(il[1])
            if iw:
                out[iw] = (q, rootleaf[rw], len(parts) > 1)
        return out


def group(items):
    """Loop argument items as one tuple text (chunked items re-parenthesized)."""
    w = [('(' + x + ')') if len(xmodel.split_top(x, ',')) > 1 else x for x in items]
    return '(' + ', '.join(w) + ', ())'


def xsem_wire(t):
    m = re.match(r'^\(?(w_\d+_(?:\d+|concat_\d+))\)?$', t.strip())
    return m.group(1) if m else None


def norm(term):
    return re.sub(r'\s+', ' ', term)


def gen(gdir, only_links=False):
    calls = [c for c in xmodel.parse_claim(gdir) if c.get('module')]
    stages = [Stage(gdir, c) for c in calls]
    stages = [s for s in stages if s.ok]
    S = stages[0].S
    imports = ['Claim'] + [f'{s.M}_Facts' for s in stages]
    out = [*(f'import {m}' for m in imports), '',
           '/-! Generated by `generator/gen_stage_links.py`: wire identities between the stages\' instance',
           'bodies from the Claim\'s stage wiring (see the generator\'s docstring). -/', '',
           'set_option maxHeartbeats 4000000', 'set_option maxRecDepth 16384', '',
           'namespace RefreshTopProof.StageLinks', '', 'open GeneratedClaim', '',
           f'/-- The members of every stage root. -/\nabbrev S : Nat := {S}', '']
    binders = '{backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel}'
    srcs, outs, srcof = {}, {}, {}
    for s in stages:
        cargs = group([norm(a) for a in s.call['args']])
        wref = lambda txt: re.sub(r'\b(w_\d+_\d+)\b', lambda m: f'({s.M}.generatedRoot.{m.group(1)} rM)', txt)
        rargs = wref(group(s.L['args']))
        routs = wref(group(s.L['outs']))
        rctx = f'{s.M}.generatedRoot.Ctx backend hashModel {s.params}'
        out.append(f'/-- The Claim\'s input tuple of `{s.M}`. -/\nabbrev {s.M}.claimArgs (external : ExternalInputs) (execution : Execution) :=\n  {cargs}\n')
        out.append(f'/-- The instance input tuple of `{s.M}` at member `i`. -/\nabbrev {s.M}.rootArgs {binders} (rM : {rctx}) (i : Fin S) :=\n  {rargs}\n')
        out.append(f'/-- The instance output tuple of `{s.M}` at member `i`. -/\nabbrev {s.M}.rootOuts {binders} (rM : {rctx}) (i : Fin S) :=\n  {routs}\n')
        head = (f'{binders} {{external : ExternalInputs}} {{execution : Execution}} {{i : Fin S}}\n'
                f'    {{rM : {rctx}}} {{cM : {s.M}.{s.inst}.Ctx backend hashModel {s.params} (i : Nat)}}')
        for w, (pi, pr, term, idx) in s.sources().items():
            src = f'({term}){" i" if idx else ""}'
            name = f'src_{s.M}_{w}'
            srcs.setdefault(norm(src), []).append((s, w, name))
            srcof[(s.M, w)] = (name, src)
            out.append(f'''theorem {name} {head}
    (hin : cM.inputs = {s.M}.rootArgs rM i) (hroot : rM.inputs = {s.M}.claimArgs external execution) :
    {s.M}.{s.inst}.{w} cM = {src} :=
  (congrArg (fun x ↦ x{pi}) hin).trans (congrArg (fun x ↦ (x{pr}){" i" if idx else ""}) hroot)
''')
        so = s.stage_out
        for w, (q, po, idx) in s.outputs().items():
            term = f'({so}{po}){" i" if idx else ""}'
            name = f'out_{s.M}_{w}'
            outs.setdefault(norm(term), []).append((s, w, name))
            out.append(f'''theorem {name} {head.replace(' {external : ExternalInputs}', '')}
    (hout : {so} = {s.M}.generatedRoot.outs rM) (hinst : {s.M}.rootOuts rM i = {s.M}.{s.inst}.outs cM) :
    {s.M}.{s.inst}.{w} cM = {term} :=
  ((congrArg (fun x ↦ (x{po}){" i" if idx else ""}) hout).trans (congrArg (fun x ↦ x{q}) hinst)).symm
''')
    # links
    nlinks = 0
    exts = set()
    def ctx_binders(s, tag):
        return (f'{{r{tag} : {s.M}.generatedRoot.Ctx backend hashModel {s.params}}} '
                f'{{c{tag} : {s.M}.{s.inst}.Ctx backend hashModel {s.params} (i : Nat)}}')
    for term, ins in srcs.items():
        for (sb, wb, nb) in ins:
            for (sa, wa, na) in outs.get(term, []):
                if sa.M == sb.M:
                    continue
                name = f'link_{sa.M}_{wa}_{sb.M}_{wb}'
                out.append(f'''/-- `{sb.M}.{wb}` is `{sa.M}`'s output `{wa}`. -/
theorem {name} {binders} {{external : ExternalInputs}} {{execution : Execution}} {{i : Fin S}}
    {ctx_binders(sa, 'A')} {ctx_binders(sb, 'B')}
    (houtA : {sa.stage_out} = {sa.M}.generatedRoot.outs rA) (hinstA : {sa.M}.rootOuts rA i = {sa.M}.{sa.inst}.outs cA)
    (hinB : cB.inputs = {sb.M}.rootArgs rB i) (hrootB : rB.inputs = {sb.M}.claimArgs external execution) :
    {sb.M}.{sb.inst}.{wb} cB = {sa.M}.{sa.inst}.{wa} cA :=
  ({nb} hinB hrootB).trans ({na} houtA hinstA).symm
''')
                nlinks += 1
                if (sa.M, wa) in srcof and (sb.M, wb) not in exts:
                    exts.add((sb.M, wb))
                    sna, ssrc = srcof[(sa.M, wa)]
                    out.append(f'''/-- `{sb.M}.{wb}` is `{sa.M}`'s pass-through output `{wa}`: the Claim value `{ssrc}`. -/
theorem ext_{sb.M}_{wb} {binders} {{external : ExternalInputs}} {{execution : Execution}} {{i : Fin S}}
    {ctx_binders(sa, 'A')} {ctx_binders(sb, 'B')}
    (hinA : cA.inputs = {sa.M}.rootArgs rA i) (hrootA : rA.inputs = {sa.M}.claimArgs external execution)
    (houtA : {sa.stage_out} = {sa.M}.generatedRoot.outs rA) (hinstA : {sa.M}.rootOuts rA i = {sa.M}.{sa.inst}.outs cA)
    (hinB : cB.inputs = {sb.M}.rootArgs rB i) (hrootB : rB.inputs = {sb.M}.claimArgs external execution) :
    {sb.M}.{sb.inst}.{wb} cB = {ssrc} :=
  ({name} houtA hinstA hinB hrootB).trans ({sna} hinA hrootA)
''')
        seen = set()
        for x, (sa, wa, na) in enumerate(ins):
            for (sb, wb, nb) in ins[x + 1:]:
                if sa.M == sb.M or (sa.M, sb.M) in seen and False:
                    continue
                name = f'link_{sa.M}_{wa}_{sb.M}_{wb}'
                out.append(f'''/-- `{sa.M}.{wa}` and `{sb.M}.{wb}` read the same Claim value `{term}`. -/
theorem {name} {binders} {{external : ExternalInputs}} {{execution : Execution}} {{i : Fin S}}
    {ctx_binders(sa, 'A')} {ctx_binders(sb, 'B')}
    (hinA : cA.inputs = {sa.M}.rootArgs rA i) (hrootA : rA.inputs = {sa.M}.claimArgs external execution)
    (hinB : cB.inputs = {sb.M}.rootArgs rB i) (hrootB : rB.inputs = {sb.M}.claimArgs external execution) :
    {sa.M}.{sa.inst}.{wa} cA = {sb.M}.{sb.inst}.{wb} cB :=
  ({na} hinA hrootA).trans ({nb} hinB hrootB).symm
''')
                nlinks += 1
    out.append('end RefreshTopProof.StageLinks\n')
    return '\n'.join(out), nlinks


if __name__ == '__main__':
    gdir, path = sys.argv[1], sys.argv[2]
    txt, n = gen(gdir, '--only-links' in sys.argv)
    open(path, 'w').write(txt)
    print(f'{path}: {len(re.findall(r"^theorem ", txt, re.M))} theorems, {n} links')
