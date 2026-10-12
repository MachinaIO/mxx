"""Views of the merged stages in the shape of the old per-part stages, and the way back.

A merged stage `Stage_<phase>` runs its body `parallel_generatedRoot_K` at every instance `i0`;
the body and its children are the old per-part root and children with the extra leading binder
`i_0`. `make_views(d)` writes `view/Stage_<phase>_I.lean` where the merged root is `mergedRoot`,
the body is `generatedRoot`, a child `X_parallel_generatedRoot_K_n` is `X_generatedRoot_n`, the
instance index is `i0`, the other loop indices are shifted down (`i_k` for `i_(k+1)`) and the
instance tables are qualified (`Stage_<phase>.sel_k`), so the generators read it as an old stage
file of part `I`. `realize(text)` maps a generated proof back onto the merged stages: names,
imports and an `i0` argument after the parameters of every application of a body scope.
"""
import os
import re

PHASES = ('preprocess', 'encrypt', 'online_eval')


def body_id(text):
    return re.search(r'∀ i : Fin \d+, parallel_generatedRoot_(\d+) ', text).group(1)


def body_ids(d):
    return {p: body_id(open(f'{d}/generated/Stage_{p}.lean').read()) for p in PHASES}


def make_view(d, phase):
    text = open(f'{d}/generated/Stage_{phase}.lean').read()
    k = body_id(text)
    ns = f'Stage_{phase}'
    text = re.sub(r'\bnamespace Stage_(\w+)\b', r'namespace Stage_\1_I', text)
    text = re.sub(r'\bend Stage_(\w+)\b', r'end Stage_\1_I', text)
    text = re.sub(r'(?<![\w.])generatedRoot\b(?!_)', 'mergedRoot', text)
    text = re.sub(rf'\b((?:parallel_|sequential_)*)parallel_generatedRoot_{k}(?![0-9])',
                  lambda m: m.group(1) + 'generatedRoot', text)
    text = old_numbering(text, phase)
    text = re.sub(r'\bi_(\d+)\b', lambda m: 'i0' if m.group(1) == '0' else f'i_{int(m.group(1)) - 1}',
                  text)
    text = re.sub(r'(?<![\w.])((?:sel|mod)_\d+)\b', rf'{ns}.\1', text)
    text = text.replace(f'{ns}.{ns}.', f'{ns}.')
    # The instance's moduli (`R`, `qc`, `Mx` of `TopTables.lean`) as single tokens.
    source = open(f'{d}/generated/Stage_{phase}.lean').read()
    tables = dict(re.findall(r'^abbrev (sel_\d+) \(i : Int\) : Int := MxxIR\.selectInt i \[([^\]]*)\]',
                             source, re.M))
    mods = dict(re.findall(r'^abbrev (mod_\d+) \(i : Int\) : Nat := \((.*)\)\.toNat$', source, re.M))
    for token, form, table in INSTANCE_MODULI:
        for sel, values in tables.items():
            if values.replace(' ', '') == table:
                for mod, body in mods.items():
                    if body == form.format(sel=sel):
                        text = text.replace(f'({ns}.{mod} ((Int.ofNat i0)))', token)
    # The scalar instance constants (`gen_tables.CONSTANTS`) as single tokens.
    from gen_tables import CONSTANTS
    for name, token, _, values in CONSTANTS:
        for sel, table in tables.items():
            if [int(x) for x in table.split(',')] == values:
                text = text.replace(f'{ns}.{sel} ((Int.ofNat i0))', token)
        text = re.sub(rf'\(\(({token})\)\)', r'(\1)', text)
        text = re.sub(rf'\(\(({token})\)\)', r'(\1)', text)
    # applications of body scopes as in the old stages (`realize` restores the instance)
    text = text.replace(' params i0 ', ' params ')
    # the instance binder of a body scope that does not use it
    text = text.replace('(params : Params) (_ : Nat) ', '(params : Params) (i0 : Nat) ')
    text = plain_families(text)
    os.makedirs(f'{d}/view', exist_ok=True)
    open(f'{d}/view/Stage_{phase}_I.lean', 'w').write(text)


# Instance moduli: token, exported form, table (the CRT primes q0 q1 q2 q3 of Q0).
_q = [1073741441, 1073741329, 1073740609, 1073740529]
INSTANCE_MODULI = [
    ('Ri0', '(1073741441) * (1073741329) * (({sel} (i)))',
     ','.join(map(str, [_q[2]] * 6 + [_q[3]] * 2))),
    ('Qci0', '(({sel} (i)))',
     ','.join(map(str, [_q[0]] * 2 + [_q[1]] * 2 + [_q[2]] * 2 + [_q[3]] * 2))),
    ('Mxi0', '(1073741441) * (({sel} (i)))',
     ','.join(map(str, [_q[1]] * 4 + [_q[2]] * 2 + [_q[3]] * 2))),
]
TOKENS = {'Ri0': '(R i0)', 'Qci0': '(qc i0)', 'Mxi0': '(Mx i0)'}
import gen_tables  # noqa: E402
# a scalar constant stands where the stage computes with its (integer) table
TOKENS.update({token: f'({name}Z (i0 : Int))' for name, token, _, _ in gen_tables.CONSTANTS})
# ... except under an explicit natural-number cast, where it is the natural number
NAT_CASTS = {f'(({name}Z (i0 : Int)) : Nat)': f'(({name} i0) : Nat)' for name, _, _, _ in gen_tables.CONSTANTS}


def _close(text, i):
    """The index of the parenthesis closing the one at `i`."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '(':
            depth += 1
        elif text[j] == ')':
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('unbalanced')


def plain_families(text):
    """`((mK : Fin n) → T)` with `T` free of `mK` as `Fin n → T` (definitionally the same), and the
    parentheses around a whole `let` type dropped, as the old exports wrote family types."""
    head = re.compile(r'\(\((m\d+) : Fin (\d+)\) → ')
    out, pos = [], 0
    for m in head.finditer(text):
        if m.start() < pos:
            continue
        end = _close(text, m.start())
        body = text[m.end():end]
        if re.search(rf'\b{m.group(1)}\b', body):
            continue
        out.append(text[pos:m.start()])
        out.append(f'(Fin {m.group(2)} → {plain_families(body)})')
        pos = end + 1
    out.append(text[pos:])
    text = ''.join(out)
    return re.sub(r'^(\s*let \S+ : )\((Fin \d+ → [^\n]*)\)( := )', lambda m: m.group(1) + m.group(2) + m.group(3)
                  if _balanced(m.group(2)) else m.group(0), text, flags=re.M)


def _balanced(s):
    depth = 0
    for ch in s:
        depth += (ch == '(') - (ch == ')')
        if depth < 0:
            return False
    return depth == 0


def node_maps(phase):
    """The body's node ids as the old part-`0_0` stage's (`maps/<phase>.json` maps old to new;
    nodes the merge added get ids from 10000)."""
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    nodes = {int(k): int(v) for k, v in json.load(open(f'{here}/maps/{phase}.json'))['nodes'].items()}
    return {new: old for old, new in nodes.items()}


def old_numbering(text, phase):
    """Rename the body's wires and child scopes to the old numbering, so the generators' node ids
    are the old stage's. Lets and atoms keep their order, so a proof destructuring the view by
    position destructures the real body; `realize` maps the scope names back."""
    back = node_maps(phase)
    if not back:
        return text
    old = lambda n: back.get(n, 10000 + n)
    # the body's own declarations (record, constraint fragments, body, relation)
    decl = re.compile(r'^(?:structure|abbrev|def) generatedRoot(?:\.Witness|\.constraints_\d+|\.body)? ', re.M)
    starts = [m.start() for m in re.finditer(r'^(?:structure|abbrev|def|theorem|/--|set_option|namespace|end) ', text, re.M)]
    starts.append(len(text))
    out, pos = [], 0
    for a, b in zip(starts, starts[1:]):
        block = text[a:b]
        if decl.match(block):
            block = re.sub(r'\bw_(\d+)_(\d+|concat_\d+)\b', lambda m: f'w_{old(int(m.group(1)))}_{m.group(2)}', block)
        out.append(text[pos:a])
        out.append(block)
        pos = b
    out.append(text[pos:])
    text = ''.join(out)
    # child scopes: the body-level node id after `generatedRoot_`
    return re.sub(r'\b((?:parallel_|sequential_)+generatedRoot_)(\d+)', lambda m: m.group(1) + str(old(int(m.group(2)))), text)


def make_views(d):
    for p in PHASES:
        make_view(d, p)


DECL = re.compile(r'^(?:@\[[^\]]*\]\s*)?(?:noncomputable )?(?:theorem|lemma|def|abbrev) (\S+)', re.M)


def bind_instance(text):
    """Add `{i0 : Nat}` to every theorem that mentions `i0` without binding it."""
    starts = [m.start() for m in DECL.finditer(text)] + [len(text)]
    out = [text[:starts[0]]]
    for a, b in zip(starts, starts[1:]):
        decl = text[a:b]
        d = re.match(r'((?:@\[[^\]]*\]\s*)?(?:noncomputable )?(?:def|abbrev) \S+)', decl)
        if d and re.search(r'\bi0\b', decl) and not re.search(r'[({]i0 : Nat[)}]', decl):
            decl = d.group(1) + ' (i0 : Nat)' + decl[d.end():]
        m = re.match(r'((?:@\[[^\]]*\]\s*)?(?:noncomputable )?theorem \S+)', decl)
        if m and re.search(r'\bi0\b', decl) and not re.search(r'[({]i0 : Nat[)}]', decl):
            # numeric side conditions may concern the instance's moduli
            decl = decl.replace('(by norm_num)', '(by inst_num)')
            fact = ' [Fact (i0 < 8)]' if 'Fact (i0 < 8)' not in decl else ''
            decl = m.group(1) + ' {i0 : Nat}' + fact + decl[m.end():]
        out.append(decl)
    return ''.join(out)


def instance_defs(texts):
    """The definitions generated with the leading binder `(i0 : Nat)`."""
    names = set()
    for text in texts:
        names |= set(re.findall(r'^(?:noncomputable )?(?:def|abbrev) (\S+) \(i0 : Nat\)', text, re.M))
    return names


def apply_instance(text, names):
    """Apply every definition of `names` to `i0` where it is used bare (as a term, not as a name in
    a `simp`/`rw`/`unfold` list)."""
    for name in names:
        short = name.rsplit('.', 1)[-1]
        pattern = re.compile(rf'(?<![\w.]){re.escape(short)}\b(?! \(i0 : Nat\))(?! i0\b)')

        def term(m):
            line_start = text.rfind('\n', 0, m.start()) + 1
            line = text[line_start:text.find('\n', m.end())]
            before = text[line_start:m.start()].rstrip()
            after = text[m.end():m.end() + 1]
            if re.search(r'\b(simp|simp only|dsimp|rw|unfold|simp_rw)\b', line) and \
                    (before.endswith('[') or before.endswith(',') or before.endswith('unfold')) and \
                    after in (',', ']', ' ', '\n', ';', ')'):
                return m.group(0)
            return f'({short} i0)'
        text = pattern.sub(term, text)
        # the declaration itself
        text = re.sub(rf'^((?:noncomputable )?(?:def|abbrev)) \({re.escape(short)} i0\) ', rf'\1 {short} ', text,
                      flags=re.M)
    return text


def instance_names(text):
    """Chain `0`'s names in generator text as the instance's: the decoding ring `Q1` as `R i0`, its
    dropped prime `q3` (outside basis lists) as `qd i0`, `q0_eq` as `q0_eqR`, and its layout
    arguments as those of `layoutR`."""
    text = re.sub(r'\bQ1\b', '(R i0)', text)
    # chain `0`'s constants (`chains.py`) as the instance's
    for value, term in (('928453219666515100085511901', '(inv4 i0)'),
                        ('618968813111010066723674601', '(half i0)'),
                        ('1152919668508929361', '(Qq i0)'), ('339699808396096934', '(cp i0)'),
                        ('536870721', '(hm i0)')):
        text = re.sub(rf'\b{value}\b', term, text)
    text = re.sub(r'\bM01\b', '(Mx i0)', text)
    # basis lists (of the instance-independent rings) keep their primes
    lists = []

    def protect(m):
        lists.append(m.group(0))
        return f'@BASIS{len(lists) - 1}@'
    text = re.sub(r'\[(?:[qp]\d, )+[qp]\d\]', protect, text)
    text = re.sub(r'\bq3\b', '(qd i0)', text)
    text = re.sub(r'@BASIS(\d+)@', lambda m: lists[int(m.group(1))], text)
    text = re.sub(r'\bq0_eq\b', '(q0_eqR i0 hi0)', text)
    import gen_luts
    from layouts import layouts
    return gen_luts.inst_layout(text, 'Backend.layout4', 'layoutQ1')


def realize(text, d):
    ids = body_ids(d)
    for p, k in ids.items():
        scope = rf'\bStage_{p}_I\.((?:parallel_|sequential_)*)generatedRoot((?:_\d+)*)(?![\w])'
        # applications of the body and its children take the instance after the parameters
        text = re.sub(scope + r'((?:\.body)?)\s+(Backend\.backend|backend)\s+(\S+)\s+(\S+)\s',
                      lambda m: m.group(0) + 'i0 ', text)
        text = re.sub(rf'\bStage_{p}_I\.mergedRoot\b', f'Stage_{p}.generatedRoot', text)
        forward = {old: new for new, old in node_maps(p).items()}

        def scope_name(m):
            suffix = m.group(2)
            if suffix:
                first, rest = suffix[1:].split('_', 1) if '_' in suffix[1:] else (suffix[1:], '')
                suffix = f'_{forward.get(int(first), int(first))}' + (f'_{rest}' if rest else '')
            return f'Stage_{p}.' + m.group(1) + f'parallel_generatedRoot_{k}' + suffix
        text = re.sub(scope, scope_name, text)
        text = re.sub(rf'\bStage_{p}_I\b', f'Stage_{p}', text)
    for token, term in TOKENS.items():
        text = re.sub(rf'\b{token}\b', term, text)
    for z, n in NAT_CASTS.items():
        text = text.replace(z, n)
    # switching facts proven with the stages' tables, used with the natural numbers
    text = re.sub(r'\bexact (hs \d)$', r'cast_exact (\1)', text, flags=re.M)
    # the inverse of a LUT coefficient is a natural number
    text = re.sub(r'(lutCoefficient2? (?:\(R i0\)|\S+) )\((\w+)Z \(i0 : Int\)\)', r'\1(\2 i0)', text)
    text = instance_names(text)
    if re.search(r'\bhi0\b|\binst_num\b|\bi0\b', text) and \
            not re.search(r'^import TopInst$', text, re.M):
        text = 'import TopInst\n' + text
    return bind_instance(text)
