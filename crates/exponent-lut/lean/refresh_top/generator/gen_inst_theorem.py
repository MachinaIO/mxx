"""Write `Part_I_Assembly.lean`: the theorem of instance `i0` of the merged stages over the bodies of
the preprocessing, encryption and online evaluation at the instance (`StageWrappers`), from
`gen_part_theorem.py`'s assembly of a part: the row is `jr i0`, the chain's constants and rings are
the instance's (`TopTables`, `TopInst`), and the link is `link_any`.

The roots are destructured (the preprocessing's names prefixed `p`, the encryption's `q`, see
`chain.prefixed`); every later step applies a theorem proven on its own (segment and chain
theorems, the root inputs, the key links) to the destructured atoms by name, so the part theorem
only wires them together. The root inputs that come from the external inputs are stated as
hypotheses on the external inputs (`hx<k>`, `hsel<k>`, `hz<k>`, `hsum<k>`, `hlc`, `hlr`, `hle`).

Usage: `python3 gen_part_theorem.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import binder_groups, group_names, prefixed
from gen_part import parts_of
from chains import chain_of, d3_subst
from patterns import destruct_root, parse
from segment import abbreviations
from stage_map import input_sources, outputs_of
from claim import decode_externals
from inst_lines import body_lines, body_projection
from wiring import _leaves, root_inputs

d = sys.argv[1]
claim = open(f'{d}/generated/Claim.lean').read()
PFX0 = {'preprocess': 'p', 'encrypt': 'q', 'online_eval': ''}
ATOM = re.compile(r"^[({](p|q)?(h\d+_\d+(?:_\d+)?|e_\d+_(?:\d+|concat_\d+)) :")
GADGETS = [('GQ7_14', 'gadgetQ7', None), ('GQ6_12', 'gadgetQ6', None), ('GQ5_10', 'gadgetQ5', None),
           ('GQ0_8', 'gadgetQ0', None), ('GQ1_6', 'gadgetQ1', None), ('GM01_4', 'gadgetM01', None),
           ('RQ7_12', 'Q7', 'GQ6_12'), ('RQ6_10', 'Q6', 'GQ5_10'), ('RQ5_8', 'Q5', 'GQ0_8'),
           ('RQ0_6', 'Q0', 'GQ1_6'), ('RQ1_4', 'Q1', 'GM01_4')]


LINES, NUMS = body_lines(d)


def view_line(stage):
    """The body relation of `stage` at the instance, in the view's names (`realize` restores the
    merged body and the instance argument)."""
    k = re.search(r'Stage_\w+\.parallel_generatedRoot_(\d+) ', LINES[stage]).group(1)
    return LINES[stage].replace(f'Stage_{stage}.parallel_generatedRoot_{k} Backend.backend model '
                                f'GeneratedClaim.stage_{NUMS[stage]}_params i0 ',
                                f'Stage_{stage}_I.generatedRoot Backend.backend model '
                                f'GeneratedClaim.stage_{NUMS[stage]}_params ')


def body_args(stage):
    """The body's input wires (view numbering) with their arguments at the instance."""
    line = view_line(stage)
    rest = line[line.index('_params ') + len('_params '):]
    depth = 0
    for k, ch in enumerate(rest):
        depth += (ch == '(') - (ch == ')')
        if depth == 0:
            break
    args = [a for a in _leaves(rest[:k + 1]) if a != '()']
    ins = root_inputs(f'{d}/generated/Stage_{stage}_I.lean')
    assert len(args) == len(ins), (stage, len(args), len(ins))
    return list(zip(ins, args))


PACKS = {}

# The instance tables of the merged stages by content: two stages' tables of the same values are
# the same constant (`Stage_x.sel_k i` unfolds to `MxxIR.selectInt i [...]`).
TABLES = {}
for _p in ('preprocess', 'encrypt', 'online_eval'):
    for _k, _v in re.findall(r'^abbrev (sel_\d+) \(i : Int\) : Int := MxxIR\.selectInt i \[([^\]]*)\]',
                             open(f'{d}/generated/Stage_{_p}.lean').read(), re.M):
        TABLES[f'Stage_{_p}.{_k}'] = _v.replace(' ', '')


def same_tables(text):
    return re.sub(r'Stage_\w+\.sel_\d+', lambda m: '[' + TABLES.get(m.group(0), m.group(0)) + ']', text)


class Externals:
    """The hypotheses on the external inputs that the part theorem states."""

    def __init__(self, part):
        # the term of every body input that comes from the external inputs (directly, or packed
        # over the instances)
        self.src = {}
        for stage, p in (('preprocess', 'pw'), ('encrypt', 'qw'), ('online_eval', 'w')):
            for w, term in body_args(stage):
                if not re.match(r'(y|out)\d', term):
                    self.src[p + w[1:]] = term
        self.hyps = {}

    @staticmethod
    def hname(x, prefix='h'):
        m = re.fullmatch(r'external\.input_(\d+)', x)
        if m:
            return prefix + m.group(1)
        # a packed argument: named by its first member
        m = re.match(r'\(?\[external\.input_(\d+)', x)
        return prefix + 'pk' + m.group(1)

    def add(self, name, ty):
        self.hyps.setdefault(name, ty)
        return name

    def value(self, name, ty):
        """The argument for semantic hypothesis `name : ty` of an input theorem, or `None`."""
        m = re.fullmatch(r'(\S+) = ([01])', ty)
        if m and m.group(1) in self.src:
            x = self.src[m.group(1)]
            return self.add(self.hname(x), f'{x} = {m.group(2)}')
        m = re.fullmatch(r'0 ≤ (\S+) ∧ \1 ≤ 1', ty)
        if m and m.group(1) in self.src:
            x = self.src[m.group(1)]
            return self.add(self.hname(x), f'0 ≤ {x} ∧ {x} ≤ 1')
        m = re.fullmatch(r'∀ i, 0 ≤ (\S+) i ∧ \1 i ≤ 1', ty)
        if m:
            x = self.src[m.group(1)]
            return self.add(self.hname(x, 'hsel'), f'∀ i, 0 ≤ {x} i ∧ {x} i ≤ 1')
        m = re.fullmatch(r'∀ i, (\S+) i = 0', ty)
        if m:
            x = self.src[m.group(1)]
            return self.add(self.hname(x, 'hz'), f'∀ i, {x} i = 0')
        m = re.fullmatch(r'∀ o : Fin (\d+), ∑ k : Fin 5, (\S+) \(cellIndex o k\) = 1', ty)
        if m:
            x = self.src[m.group(2)]
            return self.add(self.hname(x, 'hsum'),
                            f'∀ o : Fin {m.group(1)}, ∑ k : Fin 5, {x} (cellIndex o k) = 1')
        m = re.fullmatch(r'∀ k, (\S+) k \* \1 k = 1', ty)
        if m:
            x = self.src[m.group(1)]
            return self.add('hlc', f'∀ k, {x} k * {x} k = 1')
        m = re.fullmatch(r'∀ k, (\S+) k \* 2 \+ \(1 - (\S+) k\) \* 8 = \(labelExp k : Int\) \* 4', ty)
        if m:
            x, c = self.src[m.group(1)], self.src[m.group(2)]
            return self.add('hlr', f'∀ k, {x} k * 2 + (1 - {c} k) * 8 = (labelExp k : Int) * 4')
        m = re.fullmatch(r'∀ k, 0 ≤ (\S+) k ∧ \1 k ≤ 7', ty)
        if m:
            x = self.src[m.group(1)]
            return self.add('hle', f'∀ k, 0 ≤ {x} k ∧ {x} k ≤ 7')
        return None


def call(text, thm, values):
    """Apply theorem `thm` of Lean source `text`: atoms by name, `let` equations by `rfl`, every
    other explicit binder by `values(name, type)`."""
    groups, concl = binder_groups(text, thm)
    types = {}
    for g in groups:
        flat = ' '.join(g.split())
        if g.startswith('{') and ' : ' in flat:
            for n in group_names(g):
                types[n] = flat[flat.index(' : ') + 3:-1]
    args = []
    for g in groups:
        flat = ' '.join(g.split())
        if g.startswith('{'):
            args += [f'({n} := {n})' for n in group_names(g) if re.fullmatch(r'(p|q)?w_\d+_(\d+|concat_\d+)', n)]
            continue
        if not g.startswith('('):
            continue
        names = group_names(g)
        ty = flat[flat.index(' : ') + 3:-1]
        if ATOM.match(flat):
            n = names[0]
            args.append(f'({n} := rfl)' if re.match(r'(p|q)?e_', n) else f'({n} := {n})')
            continue
        for n in names:
            v = generic(n, ty, types)
            if v is None:
                v = values(n, ty)
            if v is None:
                raise ValueError(f'{thm}: no value for {n} : {ty}')
            if v != '':
                args.append(f'({n} := {v})')
    return f'{thm} (model := model) ' + ' '.join(args)


MODULI = {'Q0': 1329223801648636169536407324729467329, 'Q1': 1237937626222020133447349201,
          'D3': 1237937533988375184510142081}
INVERSES = {('Q0', 4): 996917851236477127152305493547100497, ('Q1', 4): 928453219666515100085511901,
            ('Q1', 2): 618968813111010066723674601, ('D3', 4): 928453150491281388382606561,
            ('D3', 2): 618968766994187592255071041}
WIDTH_GADGET = {14: 'hGQ7_14', 12: 'hGQ6_12', 10: 'hGQ5_10', 8: 'hGQ0_8', 6: 'hGQ1_6', 4: 'hGM01_4'}
REBASE = {('Q6', 12): 'hRQ7_12', ('Q5', 10): 'hRQ6_10', ('Q0', 8): 'hRQ5_8', ('Q1', 6): 'hRQ0_6', ('D3', 6): 'hRQ0_6',
          ('M01', 4): 'hRQ1_4', ('M2', 4): 'hRQ1_4', ('M3', 4): 'hRQ1_4', ('(R i0)', 6): 'hRQ0_6',
          ('(Mx i0)', 4): 'hRQ1_4'}


def generic(n, ty, types):
    """Arguments determined by their type: gadgets, re-encodings, inverses, the mask bounds."""
    m = re.fullmatch(r'gadgetMatrixRuns Backend\.backend 32768 (\d+) \S+', ty)
    if m:
        return WIDTH_GADGET[int(m.group(1))]
    m = re.fullmatch(r'centeredRebaseRuns (\S+) \S+', ty)
    if m and m.group(1) in types:
        t = re.fullmatch(r'ExactMatrix (\(\w+ i0\)|\S+) N 1 (\d+)', types[m.group(1)])
        return REBASE[(t.group(1).replace('RefreshTopProof.', ''), int(t.group(2)))]
    # the instance's decoding ring (`Q1` in the view's chain-`0` names, `R i0` once realized)
    m = re.fullmatch(r'\(inverse\S* : ExactPoly (?:\(R i0\)|Q1) N\) \* \(\((\d) : Nat\) : ExactPoly (?:\(R i0\)|Q1) N\) = 1', ty)
    if m:
        return {'2': '(half_mul i0 hi0)', '4': '(inv4_mul i0 hi0)'}[m.group(1)]
    m = re.fullmatch(r'\(inverse\S* : ExactPoly (\w+) N\) \* \(\((\d) : Nat\) : ExactPoly \1 N\) = 1', ty)
    if m:
        q, w = m.group(1), int(m.group(2))
        inv = INVERSES[(q, w)]
        k = (inv * w - 1) // MODULI[q]
        assert inv * w == 1 + MODULI[q] * k
        return f'(inverse_mul (k := {k}) (by norm_num))'
    if n in ('hsm', 'hr'):
        return n
    return None


def REWRITE(alias, text):
    """After every `obtain` of an input theorem's facts, rewrite the merged hash wires in them."""
    if not alias:
        return text
    rws = ', '.join(f'← hm_{w}' for w in alias)
    out = []
    for line in text.split('\n'):
        out.append(line)
        m = re.match(r'  obtain ⟨(.*)⟩ := i_\w+$', line)
        if m:
            for n in re.findall(r'[A-Za-z][\w\']*', m.group(1)):
                out.append(f'  try simp only [{rws}] at {n}')
    return '\n'.join(out)


def ALIAS(alias, text):
    if not alias:
        return text
    return re.sub(r'(?<![\w.])(p|q)?w_\d+_\d+\b', lambda m: alias.get(m.group(0), m.group(0)), text)


def atom_name(path, needle):
    _, names, atoms = parse(path, 'generatedRoot')
    hits = [n for n, a in zip(names, atoms) if needle in a]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


for part in parts_of(d):
    CH = dict(chain_of(part))
    _ab = abbreviations(open(f'{d}/TopSetup.lean').read())
    CH['QqT'] = ('RefreshTopProof.' + _ab[str(CH['Qq'])]) if str(CH['Qq']) in _ab else str(CH['Qq'])
    files = {f: open(f'{d}/Part_{part}_{f}.lean').read()
             for f in ['Inputs', 'KeyMask', 'KeySource', 'KeyFresh', 'KeySmudge']}
    ext = Externals(part)
    paths = {s: f'{d}/generated/Stage_{s}_{part}.lean' for s in ('preprocess', 'encrypt', 'online_eval')}
    lines = {stage: view_line(stage) for stage in ('preprocess', 'encrypt', 'online_eval')}
    # the broadcast payload and mask, and the instance's body outputs
    outs = sorted({int(x) for l in lines.values() for x in re.findall(r'\bout(\d+)\b', l)})
    pay, mask = outs[0], outs[1]
    ys = [NUMS[s] for s in ('preprocess', 'encrypt', 'online_eval')]

    # The online inputs and their sources: encryption outputs `qw_…` or preprocessing outputs `pw_…`.
    src = {}
    for stage, p in (('online_eval', 'w'), ('encrypt', 'qw')):
        for w, s in input_sources(d, stage, part):
            if s[0] == 'stage' and s[1].startswith('Stage_preprocess'):
                src[p + w[1:]] = 'p' + outputs_of(d, 'preprocess', part)[s[2]]
            elif s[0] == 'stage' and s[1].startswith('Stage_encrypt'):
                src[p + w[1:]] = 'q' + outputs_of(d, 'encrypt', part)[s[2]]

    # Hash samples of one tag, components and key agree across the three roots: merge them
    # (`subst`), keeping the online wire, else the preprocessing's.
    def canon(name):
        seen = set()
        while name not in seen:
            seen.add(name)
            if name in ext.src:
                return ext.src[name]
            if name in src:
                name = src[name]
                continue
            break
        return name

    HASH = re.compile(r'^MxxRuntime\.hashSample \(hashModel\) \((\[[^\]]*\])\) \((.*)\) \((w_\d+_\d+)\) (w_\d+_\d+)$')
    groups_h = {}
    for stage, p in (('online_eval', ''), ('preprocess', 'p'), ('encrypt', 'q')):
        text_s = open(paths[stage]).read()
        text_s = text_s[text_s.index('structure generatedRoot.Witness where'):]
        text_s = text_s[:text_s.index('\n\n')] + '\n'
        for n, a in zip(*parse(paths[stage], 'generatedRoot')[1:]):
            m = HASH.match(a)
            if not m:
                continue
            ty = re.search(r'\n  ' + m.group(4) + r' : ([^\n]*)\n', text_s)
            key = (m.group(1), same_tables(m.group(2)), canon(p + m.group(3)), ty.group(1) if ty else None)
            groups_h.setdefault(key, []).append((p + m.group(4), p + n))
    alias = {}
    merge = []
    for members in groups_h.values():
        if len(members) < 2:
            continue
        keep, katom = members[0]
        for w, h in members[1:]:
            merge.append(f'  have hm_{w} : {keep} = {w} := (hashSample_eq {katom}).trans (hashSample_eq {h}).symm')
            alias[w] = keep
    steps = []
    gadgets = [(n, 'gadgetMx i0 hi0' if n == 'GM01_4' else 'gadgetR i0 hi0' if n == 'GQ1_6' else
                CH['R'] if n == 'RQ1_4' else t, b) for n, t, b in GADGETS]
    for name, thm, base in gadgets:
        if base is None:
            steps.append(f'  obtain ⟨{name}, h{name}⟩ := {thm}')
        else:
            steps.append(f'  obtain ⟨{name}, h{name}⟩ := rebase_exists (p := {thm}) (by norm_num) {base}')

    def input_value(n, ty):
        if n in ('hr', 'hsm'):
            return n
        if n in ('hpay', 'qhm', 'qhp'):
            return 'rfl'
        if re.match(r'h[GR](Q\d|M01)_\d+$', n):
            return n
        return ext.value(n, ty)

    for thm in ['maskInputs', 'freshInputs'] + [f'smudgeInputs{k}' for k in range(6)] + ['sourceInputs']:
        steps.append(f'  have i_{thm} := {call(files["Inputs"], thm, input_value)}')

    # The label keys of an online key link are hash samples: online input → encryption output
    # (`encKeys`) → encryption input → preprocessing output (`preHash`).
    def hash_proof(ty):
        m = re.match(r'∀ \(k : Fin 5\) \(c : Fin 4\), !\[(\S+), (\S+), (\S+), (\S+)\] c k =', ty)
        ws = [m.group(i) for i in range(1, 5)]
        lines_ = ['by', '      intro k c']
        encs = {}
        for w in ws:
            q = src[w]
            fam = re.match(r'qw_(\d+)_\d', q).group(1)
            encs[fam] = atom_name(paths['encrypt'], f'parallel_generatedRoot_{fam} ').replace('h', 'qh', 1)
        for fam, h in encs.items():
            lines_.append(f'      obtain ⟨k{fam}_0, k{fam}_1, k{fam}_2, k{fam}_3⟩ := encKeys{fam} ({h} k)')
        lines_.append('      fin_cases c')
        for c, w in enumerate(ws):
            q = src[w]
            fam, j = re.match(r'qw_(\d+)_(\d)', q).groups()
            # Input `j` of the label encoding `fam`, as the encryption root passes it.
            atom = next(a for a in parse(paths['encrypt'], 'generatedRoot')[2]
                        if f'parallel_generatedRoot_{fam} ' in a)
            inp = re.findall(r'\(\(w_(\d+)_0 i\)', atom)[int(j)]
            pw = src[f'qw_{inp}_0']
            pfam = re.match(r'pw_(\d+)_0', pw).group(1)
            ph = atom_name(paths['preprocess'], f'parallel_generatedRoot_{pfam} ').replace('h', 'ph', 1)
            lines_.append(f'      · exact k{fam}_{j}.trans (preHash{pfam} ({ph} k))')
        return '\n'.join(lines_)

    def hash_atom(stage, w):
        hits = [n for n, a in zip(*parse(paths[stage], 'generatedRoot')[1:])
                if a.startswith('MxxRuntime.hashSample') and re.search(w + r'$', a)]
        return (PFX0[stage] + hits[0]) if hits else None

    def link_value(n, ty):
        m = re.fullmatch(r'(w_\S+) = p(w_\S+)', ty)
        if m:
            a, b = hash_atom('online_eval', m.group(1)), hash_atom('preprocess', m.group(2))
            if a and b:
                return f'((hashSample_eq {a}).trans (hashSample_eq {b}).symm)'
            # An online input: trace it to the preprocessing output it carries.
            w = m.group(1)
            while w in src or ('q' + w[1:] if w.startswith('w') else w) in src:
                w = src.get(w) or src['q' + w[1:]]
                if w.startswith('qw') and w in src:
                    continue
                break
            if w.startswith('pw') and w != 'p' + m.group(2):
                a2 = hash_atom('preprocess', w[1:])
                if a2 and b:
                    return f'((hashSample_eq {a2}).trans (hashSample_eq {b}).symm)'
        if re.fullmatch(r'\S+ = \S+', ty):
            return 'rfl'
        if n == 'haA':
            return '(' + hash_proof(ty) + ')'
        if n == 'hs':
            return 'k_freshKeyCore'
        return None

    for f, thms in [('KeyMask', ['maskKeyLink']), ('KeySource', ['sourceKeyLink']),
                    ('KeyFresh', ['freshKeyCore'] + [f'freshKeyCol{k}' for k in range(6)]),
                    ('KeySmudge', [f'smudgeKeyLink{k}' for k in range(6)])]:
        for thm in thms:
            steps.append(f'  have k_{thm} := {call(files[f], thm, link_value)}')

    # The online chains, from the facts about their inputs.
    steps.append('  obtain ⟨mOne, ⟨mEB, mHeB, mRhs⟩, ⟨mEB2, mHeB2, mRhs2⟩, mHelp5, mSw, mHelp0, mLab, '
                 '⟨mEBm, mHeBm, mRhsm⟩, ⟨mEBm2, mHeBm2, mRhsm2⟩, ⟨mSel, mHsel⟩, ⟨mSel2, mHsel2⟩, mSw1⟩ := '
                 'i_maskInputs')
    steps.append('  obtain ⟨fOne, fHelp5, fSw, fHelp0, fLab, ⟨fEB, fHeB, fRhs⟩, ⟨fEB2, fHeB2, fRhs2⟩, '
                 '⟨fSel, fHsel⟩, ⟨fSel2, fHsel2⟩, fSw1⟩ := i_freshInputs')
    for k in range(6):
        steps.append(f'  obtain ⟨sOne{k}, sHelp{k}, sLab{k}, ⟨sEB{k}, sHeB{k}, sRhs{k}⟩, '
                     f'⟨sEB2{k}, sHeB2{k}, sRhs2{k}⟩, ⟨sSel{k}, sHsel{k}⟩, ⟨sSel2{k}, sHsel2{k}⟩, sSw{k}⟩ := '
                     f'i_smudgeInputs{k}')
    steps.append('  obtain ⟨xIn, xHelp, xSw⟩ := i_sourceInputs')
    cfiles = {f: open(f'{d}/Part_{part}_{f}.lean').read() for f in ['OnlineChain', 'FreshChain', 'SmudgeChain', 'Online']}

    def chain_values(table):
        def v(n, ty):
            if n == 'hpay':
                w = re.match(r'∀ k : Fin 5, w_(\d+)_0 k', ty).group(1)
                h = atom_name(paths['online_eval'], f'parallel_generatedRoot_{w} ')
                return f'(fun k ↦ labelPay{w} k ({h} k))'
            return table.get(n)
        return v

    steps.append('  have c_mask := ' + call(cfiles['OnlineChain'], 'maskChain', chain_values({
        'hone': 'mOne', 'hrhs': 'mRhs', 'heB': 'mHeB', "hrhs'": 'mRhs2', "heB'": 'mHeB2',
        'hhelp5': 'mHelp5', 'hswitch': 'mSw', 'hhelp0': 'mHelp0', 'ha': 'mLab', 'hrhsm': 'mRhsm',
        'heBm': 'mHeBm', "hrhsm'": 'mRhsm2', "heBm'": 'mHeBm2', 'hsel': 'mHsel', "hsel'": 'mHsel2',
        'hswitch1': 'mSw1'})))
    steps.append('  have c_freshS := ' + call(cfiles['FreshChain'], 'freshSamples', chain_values({
        'hone': 'fOne', 'ha': 'fLab', 'hrhs': 'fRhs', 'heB': 'fHeB', "hrhs'": 'fRhs2', "heB'": 'fHeB2',
        'hsel': 'fHsel', "hsel'": 'fHsel2', 'hhelp5': 'fHelp5', 'hswitch': 'fSw', 'hhelp0': 'fHelp0',
        'o': ''})))
    for k in range(6):
        steps.append(f'  have c_fresh{k} := ' + call(cfiles['FreshChain'], f'freshCol{k}', chain_values({
            'hsamp': 'c_freshS', 'hswitch': 'fSw1'})))
    for k in range(6):
        steps.append(f'  have c_smudge{k} := ' + call(cfiles['SmudgeChain'], f'smudgeChain{k}', chain_values({
            'hone': f'sOne{k}', 'hhelp': f'sHelp{k}', 'ha': f'sLab{k}', 'hrhs': f'sRhs{k}', 'heB': f'sHeB{k}',
            "hrhs'": f'sRhs2{k}', "heB'": f'sHeB2{k}', 'hsel': f'sHsel{k}', "hsel'": f'sHsel2{k}',
            'hswitch': f'sSw{k}'})))
    steps.append('  have c_source := ' + call(cfiles['Online'], 'rootLut18', chain_values({
        'hinput': 'xIn', 'hhelp': 'xHelp', 'hswitch': 'xSw'})))

    # The hypotheses of the online root's output (`onlineFinal`).
    parsed = {st: parse(paths[st], 'generatedRoot') for st in paths}
    PFX = {'preprocess': 'p', 'encrypt': 'q', 'online_eval': ''}

    def find(stage, pattern):
        """The (prefixed) name and match of the unique atom of `stage` matching `pattern`."""
        hits = [(n, re.search(pattern, a)) for n, a in zip(parsed[stage][1], parsed[stage][2])]
        hits = [(n, m) for n, m in hits if m]
        assert len(hits) == 1, (stage, pattern, [n for n, _ in hits])
        n, m = hits[0]
        return PFX[stage] + n, m

    ptext = open(paths['preprocess']).read()

    def plet(w):
        m = re.search(r'let ' + w + r' [^\n]*:= (MxxRuntime\.\S+) (w_\S+) (w_\S+)\n', ptext)
        return m.group(2), m.group(3)

    def concat_chain(stage, target):
        """The five `concatColumns` atoms building `target` from six columns, and the columns."""
        names, cols = [], []
        cur = target
        for _ in range(5):
            n, m = find(stage, r'concatColumns (w_\S+) (w_\S+) ' + cur + r'$')
            names.append(n)
            cols.append(m.group(2))
            cur = m.group(1)
        cols.append(cur)
        return names[::-1], cols[::-1]

    steps.append('  obtain ⟨egI, heg, hg⟩ := RefreshTopProof.Rel.row c_mask 1')
    steps.append(f'  obtain ⟨exI, hex, hcx0⟩ := RefreshTopProof.Rel.row c_source (jr i0)')
    steps.append('  have hcx := row_assoc hcx0')
    hint = [find('encrypt', p)[0] for p in (r'modulusReduceRuns w_4_0 w_404_0$', r'modulusReduceRuns w_7_0 w_405_0$',
                                             r'concatColumns w_404_0 w_405_0 w_406_0$', r'gaussianSample \S+ \S+ w_409_0$')]
    steps.append(f'  obtain ⟨ebI, heb, hb⟩ := hint_value (sm := sm) (hw4 := rfl) (A := qw_407_0) {" ".join(hint)}')
    one = [find('encrypt', p)[0] for p in (r'modulusReduceRuns w_4_0 w_381_0$', r'modulusReduceRuns w_7_0 w_384_0$',
                                            r'gaussianSample \S+ \S+ w_402_0$')]
    g389 = find('encrypt', r'gadgetMatrixRuns backend 32768 6 w_389_0$')[0]
    steps.append(f'  obtain ⟨e1I, he1, hc1raw⟩ := one_value (sm := sm) (hw4 := rfl) {" ".join(one)}')
    steps.append(f'  have hc1 : w_334_0 = maskPart (reducePoly Q1 N sm) ![qw_395_0, qw_397_0] - GQ1_6 +\n'
                 f'      reduceMatrix Q1 N 1 6 e1I := by\n'
                 f'    rw [← gadgetMatrixRuns_unique {g389} hGQ1_6]\n    exact hc1raw')
    fnames, fcols = concat_chain('online_eval', 'w_529_0')
    steps.append(f'  obtain ⟨ewI, hew, hfw⟩ := fresh_glueQ (Qq := {CH["QqT"]}) c_fresh0 c_fresh1 c_fresh2 c_fresh3 c_fresh4 c_fresh5 '
                 + ' '.join(fnames))
    # The keys.
    FRESH = [413, 438, 460, 482, 504, 526]
    SMUDGE = [579, 634, 688, 742, 796, 850]
    kcols = ', '.join(f'fun k ↦ (blockModSwitch [q0, q1, q2, q3] 1 (![![w_{S}_0, w_{S}_1], ![w_{S}_2, w_{S}_3]] k 1) : '
                      f'ExactMatrix Q1 N 1 6) * w_417_0' for S in FRESH)
    Kw = f'(fun i ↦ rowMat (fun k ↦ (![{kcols}] : Fin 6 → Fin 2 → ExactMatrix Q1 N 1 1) k i 0 0))'
    d416 = find('online_eval', r'gadgetDecomposeRuns backend 32768 6 w_415_0 w_416_0$')[0]
    d417 = find('online_eval', r'compactCenteredRebaseRuns 16384 w_416_0 w_417_0$')[0]
    p236 = find('preprocess', r'gadgetDecomposeRuns backend 32768 6 w_235_0 w_236_0$')[0]
    p237 = find('preprocess', r'compactCenteredRebaseRuns 16384 w_236_0 w_237_0$')[0]
    steps.append(f'  have hw417 : w_417_0 = pw_237_0 := dec_rebase_unique {d416} {p236} {d417} {p237}')
    rows = []
    for i, target in enumerate(['w_319_0', 'w_357_0']):
        cn, cc = concat_chain('preprocess', target)
        entries = []
        for c, S in enumerate(FRESH):
            pre_sum = re.match(r'pw_(\d+)_1', src.get(f'w_{S}_1', '') or '')
            y, _dig = plet(cc[c])
            bms = find('preprocess', r'blockModSwitchRuns \[[^\]]*\] \(1\) (w_\S+) ' + y + '$')
            entries.append(f'(key_entry (k_freshKeyCol{c}.{i + 1}) {bms[0]} hw417)')
        rows.append(f'row_of_entries {" ".join(cn)} {" ".join(entries)}')
    steps.append(f'  have hKw : ∀ i, {Kw} i = ![pw_319_0, pw_357_0] i := by\n    intro i\n    fin_cases i\n'
                 + '\n'.join(f'    · exact {r}' for r in rows))
    steps.append('  have hKg : ∀ i, (fun k ↦ (blockModSwitch [q0, q1, q2, q3] 1 (![![w_332_0, w_332_1], '
                 '![w_332_2, w_332_3]] k 1) : ExactMatrix Q1 N 1 6)) i = ![pw_167_0, pw_338_0] i := by\n'
                 '    intro i\n    fin_cases i\n    · exact k_maskKeyLink.1\n    · exact k_maskKeyLink.2')
    steps.append(f'  have hKx : ∀ i, (fun k ↦ ![![w_52_0, w_67_0], ![w_70_0, w_73_0]] k (jr i0)) i = '
                 '![pw_186_0, pw_342_0] i := by\n'
                 '    intro i\n    fin_cases i\n    · exact k_sourceKeyLink.1\n    · exact k_sourceKeyLink.2')
    h395, h397 = find('encrypt', r'hashSample .* w_395_0$')[0], find('encrypt', r'hashSample .* w_397_0$')[0]
    h169, h339 = find('preprocess', r'hashSample .* w_169_0$')[0], find('preprocess', r'hashSample .* w_339_0$')[0]
    steps.append(f'  have hK1 : ∀ i, ![qw_395_0, qw_397_0] i = ![pw_169_0, pw_339_0] i := by\n'
                 f'    intro i\n    fin_cases i\n'
                 f'    · exact (hashSample_eq {h395}).trans (hashSample_eq {h169}).symm\n'
                 f'    · exact (hashSample_eq {h397}).trans (hashSample_eq {h339}).symm')
    # D and Y.
    a = {n: find('online_eval', p)[0] for n, p in [
        ('h338', r'hashSample .* w_338_0$'), ('h339', r'modulusReduceRuns w_338_0 w_339_0$'),
        ('h341', r'gadgetDecomposeRuns backend 32768 6 w_340_0 w_341_0$'),
        ('h342', r'compactCenteredRebaseRuns 16384 w_341_0 w_342_0$'),
        ('h344', r'gadgetMatrixRuns backend 32768 6 w_344_0$'), ('h345', r'centeredRebaseRuns w_344_0 w_345_0$'),
        ('h346', r'modulusReduceRuns w_345_0 w_346_0$'), ('h351', r'centeredRebaseRuns w_350_0 w_351_0$'),
        ('h354', r'gadgetDecomposeRuns backend 32768 4 w_353_0 w_354_0$'),
        ('h355', r'compactCenteredRebaseRuns 16384 w_354_0 w_355_0$')]}
    b = {n: find('preprocess', p)[0] for n, p in [
        ('h173', r'hashSample .* w_173_0$'), ('h174', r'modulusReduceRuns w_173_0 w_174_0$'),
        ('h176', r'gadgetDecomposeRuns backend 32768 6 w_175_0 w_176_0$'),
        ('h177', r'compactCenteredRebaseRuns 16384 w_176_0 w_177_0$'),
        ('h187', r'gadgetMatrixRuns backend 32768 6 w_187_0$'), ('h188', r'centeredRebaseRuns w_187_0 w_188_0$'),
        ('h189', r'modulusReduceRuns w_188_0 w_189_0$'), ('h194', r'centeredRebaseRuns w_193_0 w_194_0$'),
        ('h197', r'gadgetDecomposeRuns backend 32768 4 w_196_0 w_197_0$'),
        ('h198', r'compactCenteredRebaseRuns 16384 w_197_0 w_198_0$')]}
    steps.append(f'  have hD : w_342_0 = pw_177_0 := d_unique rfl {a["h338"]} {b["h173"]} {a["h339"]} {b["h174"]} '
                 f'rfl (rfl : pw_175_0 = matrixMulScalarLeft w_336_0 pw_174_0) {a["h341"]} {b["h176"]} '
                 f'{a["h342"]} {b["h177"]}')
    steps.append(f'  have hY : w_355_0 = pw_198_0 := y_uniqueG {a["h344"]} {b["h187"]} {a["h345"]} {b["h188"]} '
                 f'{a["h346"]} {b["h189"]} rfl (rfl : pw_191_0 = matrixMulScalarRight pw_189_0 w_347_0) rfl '
                 f'(rfl : pw_193_0 = matrixMulScalarRight pw_191_0 w_349_0) {a["h351"]} {b["h194"]} rfl '
                 f'(rfl : pw_196_0 = matrixMulScalarRight pw_194_0 w_352_0) {a["h354"]} {b["h197"]} '
                 f'{a["h355"]} {b["h198"]}')
    # The preimage equation of every column.
    COLS = [85, 584, 638, 692, 746, 800]
    trap = find('preprocess', r'trapdoorSample backend')[0]
    for k, P in enumerate(COLS):
        X = src[f'w_{P}_0']
        nP, mP = find('preprocess', r'preimageRunsDispatched backend w_25_0 w_25_1 (w_\S+) \(Int\.toNat \(\d+\)\) ' + X[1:] + '$')
        nT, mT = find('preprocess', r'concatRows (w_\S+) (w_\S+) ' + mP.group(1) + '$')
        r0, r1 = mT.group(1), mT.group(2)
        s0, sm0 = plet(r0)
        s1, sm1 = plet(r1)
        ns0 = find('preprocess', r'sliceMatrix w_320_0 0 1 \d+ \d+ ' + s0 + '$')[0]
        ns1 = find('preprocess', r'sliceMatrix w_358_0 0 1 \d+ \d+ ' + s1 + '$')[0]
        S = SMUDGE[k]
        link = re.search(rf'w_{S}_1 = (pw_\S+) ∧\s+w_{S}_3 = (pw_\S+)', files['KeySmudge'])
        assert (link.group(1), link.group(2)) == ('p' + sm0, 'p' + sm1), (k, link.groups(), sm0, sm1)
        steps.append(f'  have hKsm{k} : ∀ i, (fun k ↦ ![![w_{S}_0, w_{S}_1], ![w_{S}_2, w_{S}_3]] k 1) i = '
                     f'![p{sm0}, p{sm1}] i := by\n    intro i\n    fin_cases i\n'
                     f'    · exact k_smudgeKeyLink{k}.1\n    · exact k_smudgeKeyLink{k}.2')
        steps.append(f'  have pc{k} := preimage_column {trap} {nP} {nT} (by decide) {find("preprocess", r"concatColumns w_167_0 w_169_0 w_170_0$")[0]} '
                     f'{find("preprocess", r"concatColumns w_338_0 w_339_0 w_340_0$")[0]} rfl rfl {ns0} {ns1} rfl rfl '
                     f'(B := fun i ↦ rowOf qw_407_0 i) (fun i ↦ rfl) hKg hK1 hKx hKw (hKsm{k})')
    total = open(f'{d}/Part_{part}_SmudgeTotal.lean').read()
    weights = re.findall(r'noncomputable def smudgeValue(\d) [^\n]*\n[^\n]*\n\s*\(\(\(smudgeWeight(\d+)', total)
    weight = {int(k): w for k, w in weights}
    for k in range(6):
        steps.append(f'  obtain ⟨esm{k}, hesm{k}, hsm{k}⟩ := RefreshTopProof.Rel.row c_smudge{k} 1')
        steps.append(f'  obtain ⟨M{k}, hM{k}, hV{k}⟩ := smudge_glue smudgeWeight{weight[k]} rfl (smudgeWeight{weight[k]}_bound i0) '
                     f'(fun o ↦ labelExp (sSel2{k} o) + labelExp (sSel{k} o)) r (H := w_582_0) rfl')
        steps.append(f'  obtain ⟨KI{k}, hK{k}, hKb{k}⟩ := pc{k}.2')
        steps.append(f'  have hpre{k} := pc{k}.1')
        steps.append(f'  rw [← hD, ← hY] at hpre{k}')
    final = open(f'{d}/Part_{part}_OnlineFinal.lean').read()

    def final_value(n, ty):
        if n in ('heg', 'he1', 'hew', 'hex', 'heb', 'hg', 'hfw', 'hb') or re.fullmatch(
                r'h(K|Kb|pre|esm|M)\d', n):
            return n
        if n == 'hc₁':
            return 'hc1'
        m = re.fullmatch(r'hsm(\d)', n)
        if m:
            return n
        m = re.fullmatch(r'hV(\d)', n)
        if m:
            return f'(by unfold smudgeValue{m.group(1)}; exact hV{m.group(1)})'
        if re.fullmatch(r'hnoise\d', n):
            # the bound depends on the instance: check it at every instance
            return ('(lt_of_lt_of_le (by have hj : i0 < 8 := hi0; generalize i0 = j at hj ⊢; revert j; decide) '
                    '(Qq_ge i0 hi0))')
        if n == 'hcx':
            # the online root reads the source row of the instance through a merge-inserted select
            sel = find('online_eval', r'select \S+ \[w_78_0, w_83_0, w_78_0, w_83_0')[0]
            return f'((select_row (i0 := i0) rfl {sel}).trans hcx)'
        return None

    # `onlineFinal` leaves the online stage's parameters implicit and unconstrained.
    on_params = re.search(r'GeneratedClaim\.stage_\d+_params', lines['online_eval']).group(0)
    steps.append('  have hres := ' + call(final, 'onlineFinal', final_value).replace(
        'onlineFinal ', f'onlineFinal (params := {on_params}) ', 1))
    # The decode stage's residues: its sign, key and message are broadcast external inputs.
    X = decode_externals(d)
    hx = {r: 'h' + X[r][6:] for r in ('e', 'c', 'ep')}
    # the source's exponent at the instance (its hypothesis comes from the source's inputs)
    e_term = next(x for x in ext.src.values() if re.fullmatch(rf'external\.{X["e"]}|\(\[external\.{X["e"]}[,\]].*', x))
    hx['e'] = ext.hname(e_term)
    ext.add(hx['c'], f'external.{X["c"]} = 1')
    ext.add(hx['ep'], f'external.{X["ep"]} = 1 - {e_term}')
    sel = {w: ext.src[w] for w in ('pw_32_0', 'pw_37_0', 'pw_207_0', 'pw_212_0')}
    key = ext.src['w_337_0']
    sigma = f'(sigmaOf {sel["pw_32_0"]} {sel["pw_37_0"]})'
    fresh = f'(freshE {sel["pw_207_0"]} {sel["pw_212_0"]})'
    # the payload multiplies component `0`; component `1` carries the message alone
    rI = '(![r, (1 : ErrorPoly N)] (jr i0))'
    # The link's implicit wires and scalars are explicit and its side facts are separate `have`s:
    # inferring them from `rfl` equations assigns fresh terms that `hres` then has to unfold. The
    # fresh rows' equation is proved per column with the column index given: inferring it from
    # `⟨i + 8 c, _⟩` under the sum does not terminate.
    R, qc = CH['R'], CH['qc']
    wires = (f'(R := {R}) (qc := {qc}) (A := w_338_0) (A\' := w_339_0) (GR := w_344_0) '
             f'(w345 := w_345_0) (w346 := w_346_0) (k3 := {CH["k3"]}) (k1 := {CH["k1"]}) (w347 := w_347_0) '
             f'(w349 := w_349_0) (w348 := w_348_0) (w350 := w_350_0) (res := w_855_0)')
    k = f"actionConst' {CH['cp']} {CH['kdiv']}"
    steps.append(f'''  have hβ := mask_sigmaC (qc := {qc}) (h := {CH["h"]}) (hm_two i0 hi0) mHsel mHsel2 r
  have hσs := sigmaPoly_eq {sigma}
  have hxs := source_message {hx["e"]} {hx["ep"]} {hx["c"]} {rI}
  have hR : 1 < {R} := R_gt_one i0 hi0
  have hq : 1 < {qc} := qc_gt_one i0 hi0
  have hdR : {qc} ∣ {R} := qc_dvd_R i0 hi0
  have hd0 : {qc} ∣ Q0 := qc_dvd_Q0 i0 hi0
  have e347 : w_347_0 = (matrixPolynomial [{CH["k3"]}] : ExactMatrix {qc} N 1 1) := rfl
  have e349 : w_349_0 = (matrixPolynomial [{CH["k1"]}] : ExactMatrix {qc} N 1 1) := rfl
  have e348 : w_348_0 = matrixMulScalarRight w_346_0 w_347_0 := rfl
  have e350 : w_350_0 = matrixMulScalarRight w_348_0 w_349_0 := rfl''')
    # the online stage's gadget is in `Q₁` at every instance (`link_anyG`, `link_caseQ`)
    link = (f'link_anyG (G := G) {wires} hR hq hdR hd0 h339_0 h345_0 h346_0 e347 e349 e348 e350 '
            f'(k := {k}) (link_caseQ i0 hi0 h344_0 hG) ')
    steps.append(f'  have hfin := {link}hβ hσs hxs {fresh} ?hE hres\n  case hE =>\n    funext rr cc\n    fin_cases cc'
                 + ''.join(f'\n    · exact congrArg (reducePoly {R} N) (fresh_entry fHsel fHsel2 r {c})' for c in range(6)))
    steps.append('  subst houtO')
    # `A'` under the instance's tag, and the payload of the row in the goal's form
    steps.append(f'  have hA : w_338_0 = model.sample Q0 N 2 6 {key} (completeHashTag (topTagI i0) []) :=\n'
                 f'    (hashSample_eq h338_0).trans (by rw [completeHashTag_nil]; rfl)\n'
                 f'  rw [hA, row_payload] at hfin')
    steps.append('  unfold topValue\n  exact hfin')
    concl_r = '((![fun _ _ ↦ reducePoly Q0 N r, 1] : Fin 2 → ExactMatrix Q0 N 1 1) (jr i0))'
    ds = {s: destruct_root(paths[s], f'Stage_{s}_{part}', 'generatedRoot', h)
          for s, h in (('preprocess', 'hpre'), ('encrypt', 'henc'), ('online_eval', 'hon'))}
    dpre = prefixed(ds['preprocess'], 'p').replace('hout⟩ := hpre', 'houtP⟩ := hpre')
    denc = prefixed(ds['encrypt'], 'q').replace('hout⟩ := henc', 'houtQ⟩ := henc')
    don = ds['online_eval'].replace('hout⟩ := hon', 'houtO⟩ := hon')
    # the residue: output `w_855_0` of the online body
    on_outs = outputs_of(d, 'online_eval', part)
    res = body_projection(f'y{ys[2]}', on_outs.index('w_855_0'), len(on_outs) + 1)
    hyps = '\n'.join(f'    ({n} : {t})' for n, t in ext.hyps.items())
    # the bound of a smudge LUT's weights past the first (numerals, the same at every instance)
    bounds = ''.join(f'''theorem smudgeWeight{weight[k]}_bound (i0 : Nat) :
    ∑ r : Fin 19, (smudgeWeight{weight[k]} i0 r.succ).natAbs ≤ 274877906942 := by
  unfold smudgeWeight{weight[k]}
  simp only [Matrix.cons_val_succ]
  decide

''' for k in range(6))
    body = f'''import Claim
import TopGadgets
import Part_{part}_Inputs
import Part_{part}_KeyFacts
import Part_{part}_KeyMask
import Part_{part}_KeySource
import Part_{part}_KeyFresh
import Part_{part}_KeySmudge
import Part_{part}_OnlineChain
import Part_{part}_FreshChain
import Part_{part}_SmudgeChain
import Part_{part}_Labels
import Part_{part}_OnlineFinal
import TopGlue
import TopFinal

import TopRow
import TopAssembly

/-! The assembly of instance `i0` of the merged stages (written by `generator/gen_inst_theorem.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof Mxx.ExponentLut.Refresh

{bounds}set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem partAssembly {{model : HashModel}} {{external : GeneratedClaim.ExternalInputs}} {' '.join(f'{{out{k} : ExactMatrix Q0 N 1 1}}' for k in outs)} {' '.join(f'{{y{k} : _}}' for k in ys)}
    {{r sm : ErrorPoly N}} (hr : polyNorm r ≤ 1) (hsm : polyNorm sm ≤ 1)
    (h{pay} : out{pay} = fun _ _ ↦ reducePoly Q0 N r) (h{mask} : out{mask} = fun _ _ ↦ reducePoly Q0 N sm)
{hyps}
    (hpre : {lines['preprocess']})
    (henc : {lines['encrypt']})
    (hon : {lines['online_eval']})
    {{G : ExactMatrix Q1top N 1 6}} (hG : gadgetMatrixRuns Backend.backend 32768 6 G) :
    Residue {res} (topValue model {key} (topTagI i0)
      {sigma} {concl_r} external.{X["c"]} external.{X["ep"]} G {fresh}) := by
  subst h{pay} h{mask}
{dpre}  subst houtP
{denc}  subst houtQ
{don}''' + '\n'.join(merge) + '\n' + REWRITE(alias, '\n'.join(steps)) + f'''

end RefreshTopProof.Part_{part}
'''
    if CH['R'] == 'D3':
        k = body.index(':= by\n')
        body = body[:k] + d3_subst(body[k:], d)
        body = body.replace('import TopGadgets\n', f'import TopGadgets\nimport Part_{part}_Local\n')
    open(f'{d}/Part_{part}_Assembly.lean', 'w').write(body)
    break
