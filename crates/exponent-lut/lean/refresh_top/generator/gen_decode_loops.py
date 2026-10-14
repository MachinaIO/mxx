"""Write the theorem of the looped decode stage of parameter directory `d` from the generic lemmas
of `GenDecode.lean`: the decode stage outputs zero when every CRT term it reads is the term of the
expected row of its component plus one small error row.

The decode stage (`generated/Stage_decode.lean`) reads a family `T : Fin S → M_Q(1 × m)` of CRT
terms (member `t R + j`: prime `t`, component `j`). Per component it reads member `j` and runs a
strided sequential loop over members `(i + 1) R + j`; subtracts the expected row
`σ A'_0 + A'_1 - (Δ r) (c X^e) G'` (σ from its coefficient values, `A'` the component's column
block of the caller's `A'` input, `G'` the centered rebase of the gadget); and runs a sequential loop over the
columns whose body exports the column's coefficients and runs an inner loop of rounding bits.

Every role is recognized by atom kinds, loop shapes and data flow, never by wire ids or scope
numbers: the strided loops are the matrix-state loops whose initial state is a static family
read; the column loops are the integer-state loops whose body slices a column and exports its
values; a column loop's difference is `matrixSub s Y` for a strided loop's final `s`; `Y`'s rows
are slices of the column block `m j` of the `A'` input. Nothing is enumerated
over members, columns or coefficients: the slot count, column count and ring dimension only
appear in types and in `decide` checks of `K R ≤ S`.

The output has
* per child scope an adapter to the step shapes `StrideStep`, `ColumnStep`, `RoundStep`;
* `decodeR`, `decodeValue`: the target value of component `j`;
* `<name>_gen`: the theorem for any CRT tables `qN`, `KN` of the `K` primes (hypotheses: the
  primes exceed one, `KN` is a CRT basis, the primes are pairwise coprime, their product is `Q`);
* when the online stage's CRT-term role is found in `generated/Stage_online_eval.lean` (the
  centered rebase of a residue times `matrixPolynomial [sel (i0)]`), the tables `decode_qN`,
  `decode_KN` read from its instance tables (checked to depend on `i0 / R` only), their
  certificates by `decide`, and `<name>` for them.

Usage: `python3 gen_decode_loops.py <parameter directory> [--out FILE] [--name THEOREM]
[--namespace NS] [--online FILE | --no-online]`. The default output is `<d>/TopDecodeLoops.lean`
with theorem `decode_zero_loops` in namespace `DecodeLoops`.
"""
import argparse
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('d')
ap.add_argument('--out')
ap.add_argument('--name', default='decode_zero_loops')
ap.add_argument('--namespace', default='DecodeLoops')
ap.add_argument('--online')
ap.add_argument('--no-online', action='store_true')
a = ap.parse_args()
d = a.d
out_path = a.out or f'{d}/TopDecodeLoops.lean'
path = f'{d}/generated/Stage_decode.lean'
NS = 'Stage_decode'
text = open(path).read()
lines = text.split('\n')


def fail(msg):
    sys.exit(f'gen_decode_loops: {msg}')


def check(cond, msg):
    if not cond:
        fail(msg)


# ---------------------------------------------------------------------------------------------
# Parsing the generated module

def block(head):
    """The indented lines following the line starting with `head`, and the head line."""
    for i, line in enumerate(lines):
        if line.startswith(head):
            out, j = [], i + 1
            while j < len(lines) and lines[j].startswith('  '):
                out.append(lines[j])
                j += 1
            return line, out
    return None, None


def split_top(s, sep):
    """Split `s` at the top-level (paren/bracket depth 0) occurrences of `sep`."""
    parts, depth, cur, i = [], 0, '', 0
    while i < len(s):
        c = s[i]
        if c in '([{⟨':
            depth += 1
        elif c in ')]}⟩':
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            parts.append(cur.strip())
            cur = ''
            i += len(sep)
            continue
        cur += c
        i += 1
    parts.append(cur.strip())
    return parts


def sig_params(head):
    """`name → type` of the explicit binders `(name : type)` of a declaration head."""
    out, i = {}, 0
    while True:
        i = head.find('(', i)
        if i < 0:
            return out
        depth, j = 0, i
        while True:
            if head[j] == '(':
                depth += 1
            elif head[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        inner = head[i + 1:j]
        m = re.match(r'(\S+) : (.*)$', inner, re.S)
        if m:
            out[m.group(1)] = m.group(2).strip()
        i = j + 1


def sig_list(head):
    """The explicit binders `(name : type)` of a declaration head, in order."""
    out, i = [], 0
    while True:
        i = head.find('(', i)
        if i < 0:
            return out
        depth, j = 0, i
        while True:
            if head[j] == '(':
                depth += 1
            elif head[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        m = re.match(r'(\S+) : (.*)$', head[i + 1:j], re.S)
        if m:
            out.append((m.group(1), m.group(2).strip()))
        i = j + 1


def tuple_types(t):
    """The component types of a right-nested product type ending in `Unit`."""
    parts = split_top(t, '×')
    check(parts[-1] == 'Unit', f'input tuple {t} ends in Unit')
    return parts[:-1]


def proj_pos(expr):
    """The position of `inputs.2.2.1`-style projections."""
    m = re.fullmatch(r'inputs((?:\.2)*)\.1', expr)
    return len(m.group(1)) // 2 if m else None


def nested(names):
    """The right-nested tuple literal `(a, (b, (c, ())))`."""
    out = '()'
    for x in reversed(names):
        out = f'({x}, {out})'
    return out


class Scope:
    """A generated scope: its lets, witnesses, binders and constraint atoms."""

    def __init__(self, name):
        self.name = name
        head, body = block(f'def {name} ')
        check(head is not None, f'scope {name}')
        self.head = head
        self.params = sig_params(head)
        self.inputs = tuple_types(self.params['inputs'])
        self.witness = any('∃ witness' in l for l in body)
        if self.witness:
            _, body = block(f'abbrev {name}.body ')
        self.body = body
        self.lets = {}
        for l in body:
            m = re.match(r'\s*let (w_\S+)(?: : [^\n]*?)? := (.*)$', l)
            if m:
                self.lets[m.group(1)] = m.group(2).strip()
        self.binders = 0 if self.witness else sum(len(re.findall(r'∃ \(w_\S+ :', l)) for l in body)
        # The constraint chain: the trailing `X.constraints_k ...` calls of the body, in order.
        calls = []
        for l in body:
            calls += re.findall(rf'(?<![\w.])({re.escape(name)}\.constraints_\d+) ', l)
        check(calls, f'constraints of {name}')
        self.groups = [self.atoms_of(c) for c in calls]

    def atoms_of(self, c):
        _, b = block(f'abbrev {c} ')
        out = []
        for l in b:
            atom = l.strip()
            if atom.endswith(' ∧'):
                atom = atom[:-2].strip()
            m = re.match(rf'({re.escape(self.name)}\.constraints_\d+) ', atom)
            out.append(self.atoms_of(m.group(1)) if m else atom)
        return out

    def flat(self):
        def go(g):
            for x in g:
                if isinstance(x, list):
                    yield from go(x)
                else:
                    yield x
        return list(go(self.groups))

    def pattern(self, name_of):
        """The `obtain` pattern destructuring the scope, naming atom `x` by `name_of(x)`."""
        def pat(g):
            return '⟨' + ', '.join(pat(x) if isinstance(x, list) else name_of(x) for x in g) + '⟩'
        inner = [pat(g) for g in self.groups]
        if len(inner) == 1:
            inner = [inner[0][1:-1]]
        lead = ['w'] if self.witness else [f'b{k}' for k in range(self.binders)]
        return '⟨' + ', '.join(lead + inner) + '⟩'


ITER = re.compile(r'MxxIR\.IterRuns \(fun \(i : Nat\) \(current next : (.*?)\) => (\S+) (.*?) i '
                  r'\(current, (.*)\) next\) \(Int\.toNat \((-?\d+)\)\) (\S+) (\S+)$')


def iter_info(atom):
    m = ITER.match(atom)
    if not m:
        return None
    state, scope, lead, args, count, init, final = m.groups()
    return dict(state=state, scope=scope, lead=len(split_top(lead.strip(), ' ')), args=unnest(args.strip()),
                count=int(count), init=init, final=final)


def unnest(t):
    """`(a, (b, (c, ())))` → `[a, b, c]`."""
    if t == '()':
        return []
    parts = split_top(t[1:-1], ',')
    return [parts[0]] + unnest(', '.join(parts[1:]).strip())


TRIVIAL = re.compile(r'^\(?-?\d+\)? (≤|<|=) \(?-?\d+\)?$|^0 ≤ \(\(?Int\.ofNat i_\d\)?\)$|'
                     r'^\(?\(?Int\.ofNat i_\d\)?\)? < \(?1 \+ \(?Int\.ofNat i_\d\)?\)?$|'
                     r'^\(?1 \+ \(?Int\.ofNat i_\d\)?\)? ≤ \d+$|^0 ≤ w_\S+$|^w_\S+ < \d+$|^w_\S+ ≠ 0$')

# ---------------------------------------------------------------------------------------------
# The root

root = Scope('generatedRoot')
check(root.witness, 'the root has a witness')
lets = root.lets
inputs_ty = root.params['inputs']
input_wire = {w_: proj_pos(e) for w_, e in lets.items() if proj_pos(e) is not None}
input_expr = {w_: lets[w_] for w_ in input_wire}
witness_fields = {w_ for w_, e in lets.items() if e == f'witness.{w_}'}
types = {}
for c in re.findall(r'abbrev (generatedRoot\.constraints_\d+) ', text):
    h, _ = block(f'abbrev {c} ')
    types.update(sig_params(h))
for l in root.body:
    m = re.match(r'\s*let (w_\S+) : (.*?) := ', l)
    if m:
        types[m.group(1)] = m.group(2)
MAT = re.compile(r'Mxx\.Primitives\.ExactMatrix (\S+) (\d+) (\d+) (\d+)$')
moduli = dict(re.findall(r'abbrev (modulus_\d+) : Nat := (\d+)', text))


def unabbrev(t):
    """`t` with the module's modulus abbreviations replaced by their values."""
    return re.sub(r'\b(modulus_\d+)\b', lambda mm: moduli[mm.group(1)], t)


def mat(x):
    m = MAT.match(types[x].strip())
    check(m, f'matrix type of {x}: {types.get(x)}')
    q = m.group(1)
    return (int(moduli.get(q, q)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


atoms = root.flat()
kinds = {}
for x in atoms:
    if x.startswith('outputs ='):
        k = 'out'
    elif ITER.match(x):
        k = 'iter'
    else:
        m = re.match(r'MxxRuntime\.(\w+) ', x)
        k = m.group(1) if m else ('trivial' if TRIVIAL.match(x) else None)
    check(k, f'unrecognized root atom {x}')
    kinds[x] = k


def last_wire(x):
    return re.findall(r'\bw_\d+_\d+\b', x)[-1]


def term_text(w_):
    """How wire `w_` reads after the destructuring: witnesses as `w.w_…`, lets inlined."""
    if w_ in witness_fields:
        return f'w.{w_}'
    return lets[w_]


# Child scopes, classified by shape.
loops = [iter_info(x) for x in atoms if kinds[x] == 'iter']
scopes = {}
for lp in loops:
    if lp['scope'] not in scopes:
        scopes[lp['scope']] = Scope(lp['scope'])


def is_stride(lp):
    s = scopes[lp['scope']]
    return MAT.match(lp['state']) and any(re.match(r'MxxRuntime\.familyGetDynamic ', x) for x in s.flat())


def is_column(lp):
    s = scopes[lp['scope']]
    return lp['state'] == 'Int' and any(x.startswith('MxxRuntime.sliceMatrix ') for x in s.flat()) and \
        any(x.startswith('MxxRuntime.polynomialValues false ') for x in s.flat())


strides = [lp for lp in loops if is_stride(lp)]
columns = [lp for lp in loops if is_column(lp)]
check(len(strides) + len(columns) == len(loops), 'every root loop is a strided sum or a column loop')
check(len(strides) == len(columns) and strides, 'one strided sum and one column loop per component')

# Static family reads (the strided sums' first members).
statics = {}
for x in atoms:
    if kinds[x] == 'familyGetStatic':
        m = re.match(r'MxxRuntime\.familyGetStatic (\S+) \((-?\d+)\) (\S+)$', x)
        check(m, f'static read {x}')
        statics[m.group(3)] = (m.group(1), int(m.group(2)), x)


def stride_shape(scope):
    """`(R, j)` of a strided child: its read index is `(i + 1) R + j`."""
    s = scopes[scope]
    reads = [x for x in s.flat() if x.startswith('MxxRuntime.familyGetDynamic ')]
    check(len(reads) == 1, f'{scope} reads one member')
    fam, idx, val = re.match(r'MxxRuntime\.familyGetDynamic (\S+) (\S+) (\S+)$', reads[0]).groups()
    L = s.lets

    def bin_(w_, op):
        m = re.fullmatch(rf'\((w_\S+) {re.escape(op)} (w_\S+)\)', L.get(w_, ''))
        return m.groups() if m else None

    def lit(w_):
        check(re.fullmatch(r'-?\d+', L.get(w_, '')), f'{scope}: {w_} is a literal')
        return int(L[w_])
    a_, jw = bin_(idx, '+')
    b_, Rw = bin_(a_, '*')
    iw, onew = bin_(b_, '+')
    check(re.fullmatch(r'\(Int\.ofNat i_0\)', L[iw]) and lit(onew) == 1, f'{scope}: index (i + 1) R + j')
    outs = [x for x in s.flat() if x.startswith('outputs = ')]
    check(len(outs) == 1, f'{scope}: one output')
    ow = outs[0].split()[-1]
    st = L[ow]
    check(re.fullmatch(rf'MxxRuntime\.matrixAdd (w_\S+) {re.escape(val)}', st), f'{scope}: adds the member')
    return lit(Rw), lit(jw), fam


comps = []
for lp in strides:
    R_, j_, _ = stride_shape(lp['scope'])
    check(lp['init'] in statics, f'strided sum {lp["scope"]} starts at a static read')
    fam, j0, _ = statics[lp['init']]
    check(j0 == j_, f'{lp["scope"]}: static member {j0} = offset {j_}')
    comps.append(dict(stride=lp, R=R_, j=j_, fam=fam))
R = comps[0]['R']
check(all(c['R'] == R for c in comps) and sorted(c['j'] for c in comps) == list(range(R)),
      'components 0 … R-1 with one stride R')
fam = comps[0]['fam']
check(all(c['fam'] == fam for c in comps), 'one CRT-term family')
check(fam in input_wire, 'the CRT-term family is an input')
S = int(re.match(r'\(?Fin (\d+) →', root.inputs[input_wire[fam]]).group(1))
Kc = {c['stride']['count'] + 1 for c in comps}
check(len(Kc) == 1, 'every component sums the same number of terms')
K = Kc.pop()
check(K * R <= S, f'K R = {K * R} ≤ S = {S}')


def op(w_, name):
    m = re.fullmatch(rf'MxxRuntime\.{name} (w_\S+) (w_\S+)', lets.get(w_, ''))
    return m.groups() if m else None


slices = {}
for x in atoms:
    if kinds[x] == 'sliceMatrix':
        m = re.match(r'MxxRuntime\.sliceMatrix (\S+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (\S+)$', x)
        check(m, f'slice {x}')
        slices[m.group(6)] = (m.group(1), int(m.group(2)), int(m.group(4)), x, int(m.group(3)), int(m.group(5)))
sigma_atoms = [x for x in atoms if kinds[x] == 'polynomialFromValues']
check(len(sigma_atoms) == 1, 'one σ')
bits, sigma = re.match(r'MxxRuntime\.polynomialFromValues false (\S+) (\S+)$', sigma_atoms[0]).groups()
check(bits in input_wire, 'σ is built from input coefficient values')
gad = [x for x in atoms if kinds[x] == 'gadgetMatrixRuns']
check(len(gad) == 1, 'one gadget')
gbase, gcols, gwire = re.match(r'MxxRuntime\.gadgetMatrixRuns backend (\d+) (\d+) (\S+)$', gad[0]).groups()
rebs = [x for x in atoms if kinds[x] == 'centeredRebaseRuns']
greb = [x for x in rebs if x.split()[1] == gwire]
check(len(greb) == 1 and len(rebs) == 1, 'the gadget is rebased once')
gprime = greb[0].split()[2]
check(gprime in witness_fields, 'the rebased gadget is a witness')

used_atoms = {sigma_atoms[0]: 'hσ', gad[0]: 'hgad', greb[0]: 'hreb'}
for c in comps:
    col = next((lp for lp in columns if op(lp['args'][0], 'matrixSub') and
                op(lp['args'][0], 'matrixSub')[0] == c['stride']['final']), None)
    check(col, f'the column loop of component {c["j"]} subtracts from its strided sum')
    c['col'] = col
    D = col['args'][0]
    Y = op(D, 'matrixSub')[1]
    P, M = op(Y, 'matrixSub')
    scaled, row1 = op(P, 'matrixAdd')
    sg, row0 = op(scaled, 'matrixMulScalarLeft')
    check(sg == sigma, f'component {c["j"]}: σ A0')
    check(row0 in slices and row1 in slices, f'component {c["j"]}: rows of A\'')
    h0, r0i, c0i = slices[row0][:3]
    h1, r1i, c1i = slices[row1][:3]
    check(h0 == h1 and h0 in slices and (r0i, r1i, c0i, c1i) == (0, 1, 0, 0), f'component {c["j"]}: A\' rows 0, 1')
    key, ar0, ac0, hatom, ar1, ac1 = slices[h0]
    mj = mat(h0)[3]
    check((ar0, ar1, ac0, ac1) == (0, 2, mj * c['j'], mj * c['j'] + mj),
          f'component {c["j"]}: A\' is the column block {c["j"]} of the A\' input')
    drm, gp = op(M, 'matrixMulScalarLeft')
    check(gp == gprime, f'component {c["j"]}: times G\'')
    dr, mono = op(drm, 'matrixMulScalarLeft')
    delta_w, rw = op(dr, 'matrixMulScalarLeft')
    dm = re.fullmatch(r'\(MxxRuntime\.matrixPolynomial \[\((-?\d+)\)\] : .*\)', lets[delta_w])
    check(dm, f'component {c["j"]}: Δ is a literal')
    mm = re.fullmatch(r'MxxRuntime\.multiplyMonomial (w_\S+) (w_\S+)', lets[mono])
    lifted, expo = mm.groups()
    coef = re.fullmatch(r'MxxRuntime\.liftInteger (w_\S+)', lets[lifted]).group(1)
    check(coef in input_wire and expo in input_wire and key in input_wire, 'c, e and A\' are inputs')
    if rw in input_wire:
        r_text = input_expr[rw]
    else:
        rm = re.fullmatch(r'\((-?\d+) : (Mxx\.Primitives\.ExactMatrix .*)\)', lets[rw])
        check(rm, f'component {c["j"]}: r is an input or a literal')
        r_text = f'({rm.group(1)} : ExactMatrix {mat(Y)[0]} {mat(Y)[1]} 1 1)'
    c.update(Y=Y, row0=row0, row1=row1, key=key,
             delta=int(dm.group(1)), r=r_text, coef=coef, expo=expo)
    used_atoms[slices[row0][3]] = f'hr{c["j"]}0'
    used_atoms[slices[row1][3]] = f'hr{c["j"]}1'
    used_atoms[hatom] = f'hA{c["j"]}'
    used_atoms[statics[c['stride']['init']][2]] = f'hinit{c["j"]}'
check(len({(c['key'], c['delta'], c['coef'], c['expo']) for c in comps}) == 1,
      'the components share A\', Δ, c and e')
for x in atoms:
    if kinds[x] == 'iter':
        lp = iter_info(x)
        c = next((c for c in comps if c['stride']['final'] == lp['final'] and lp in [c['stride']]), None)
        if c:
            used_atoms[x] = f'hs{c["j"]}'
        c = next((c for c in comps if c['col'] == lp), None)
        if c:
            used_atoms[x] = f'hc{c["j"]}'
outs = [x for x in atoms if kinds[x] == 'out']
check(len(outs) == 1, 'one output')
used_atoms[outs[0]] = 'hout'
for x in atoms:
    check(x in used_atoms or kinds[x] == 'trivial', f'unused root atom {x}')

Q, N, _, m = mat(comps[0]['Y'])
Qg = mat(gwire)[0]
check(int(gcols) == m, 'the gadget has the column count')

# The output: a sum of literal zeros and column-loop finals whose initial states are zeros.
def zero_sum(e):
    e = e.strip()
    if re.fullmatch(r'-?\d+', e):
        return int(e) == 0
    if e in lets and e not in witness_fields:
        return zero_sum(lets[e])
    if e.startswith('(') and e.endswith(')'):
        return all(zero_sum(p) for p in split_top(e[1:-1], '+'))
    c = next((c for c in comps if c['col']['final'] == e), None)
    return bool(c) and zero_sum(c['col']['init'])


out_w = outs[0].split()[-1]
check(zero_sum(out_w), 'the output sums the column loops started at zero')

# The column scopes and their coefficient-loop children.
col_info = {}
for c in comps:
    sc = scopes[c['col']['scope']]
    inner = [iter_info(x) for x in sc.flat() if ITER.match(x)]
    check(len(inner) == 1 and inner[0]['state'] == 'Int', f'{sc.name}: one coefficient loop')
    il = inner[0]
    if il['scope'] not in scopes:
        scopes[il['scope']] = Scope(il['scope'])
    rs = scopes[il['scope']]
    L = rs.lets
    reads = [x for x in rs.flat() if x.startswith('MxxRuntime.familyGetDynamic ')]
    check(len(reads) == 1, f'{rs.name}: one coefficient read')
    vals_w, idx, v = re.match(r'MxxRuntime\.familyGetDynamic (\S+) (\S+) (\S+)$', reads[0]).groups()
    check(L.get(idx) == '(Int.ofNat i_1)', f'{rs.name}: reads coefficient i')
    ow = next(x for x in rs.flat() if x.startswith('outputs = ')).split()[-1]
    st, md = re.fullmatch(r'\((w_\S+) \+ (w_\S+)\)', L[ow]).groups()
    num, two = re.fullmatch(r'\((w_\S+) % (w_\S+)\)', L[md]).groups()
    sm, qd = re.fullmatch(r'\((w_\S+) / (w_\S+)\)', L[num]).groups()
    dbl, hf = re.fullmatch(r'\((w_\S+) \+ (w_\S+)\)', L[sm]).groups()
    vv, tw = re.fullmatch(r'\((w_\S+) \* (w_\S+)\)', L[dbl]).groups()
    check(vv == v and L[tw] == '2' and L[two] == '2' and L[st] == 'inputs.1',
          f'{rs.name}: adds ((2 v + half) / q) % 2')
    pos = {proj_pos(L[vals_w]): 'vals', proj_pos(L[hf]): 'half', proj_pos(L[qd]): 'qc', 0: 'c'}
    check(len(pos) == 4 and None not in pos, f'{rs.name}: inputs')
    round_names = [pos.get(p, f'x{p}') for p in range(len(rs.inputs))]
    # The column scope: its difference input and the inputs it passes on as half and q.
    sl = [x for x in sc.flat() if x.startswith('MxxRuntime.sliceMatrix ')]
    check(len(sl) == 1, f'{sc.name}: one slice')
    sm_ = re.match(r'MxxRuntime\.sliceMatrix (\S+) 0 1 \(Int\.ofNat i_0\) \(1 \+ \(Int\.ofNat i_0\)\) (\S+)$', sl[0])
    check(sm_, f'{sc.name}: slices column i')
    Dw, xw = sm_.groups()
    pv = [x for x in sc.flat() if x.startswith('MxxRuntime.polynomialValues false ')]
    check(len(pv) == 1, f'{sc.name}: exports the column')
    check(pv[0].split()[2] == xw and pv[0].split()[3] == il['args'][round_names.index('vals') - 1],
          f'{sc.name}: the coefficient loop reads the exported values')
    check(il['init'] == 'current' or sc.lets.get(il['init']) == 'inputs.1', f'{sc.name}: loop from the state')
    cpos = {0: 'cur', proj_pos(sc.lets[Dw]): 'D'}
    for role in ('half', 'qc'):
        cpos[proj_pos(sc.lets[il['args'][round_names.index(role) - 1]])] = role
    check(len(cpos) == 4 and None not in cpos, f'{sc.name}: inputs')
    col_names = [cpos.get(p, f'x{p}') for p in range(len(sc.inputs))]
    ocol = next(x for x in sc.flat() if x.startswith('outputs = ')).split()[-1]
    check(ocol == il['final'], f'{sc.name}: outputs the coefficient loop')
    col_info[sc.name] = dict(round=rs.name, round_names=round_names, col_names=col_names, rlead=il['lead'])
    # The root passes the half modulus and the modulus.
    c['half'] = term_text(c['col']['args'][col_names.index('half') - 1])
    c['qc'] = term_text(c['col']['args'][col_names.index('qc') - 1])

# ---------------------------------------------------------------------------------------------
# The online stage's CRT-term role, for the concrete tables.

tables = None
online = None if a.no_online else (a.online or f'{d}/generated/Stage_online_eval.lean')
if online and os.path.exists(online):
    otext = open(online).read()
    sel = {k: [int(v) for v in re.findall(r'-?\d+', t)] for k, t in
           re.findall(r'abbrev (sel_\d+) \(i : Int\) : Int := MxxIR\.selectInt i \[(.*?)\]', otext)}
    mods = dict(re.findall(r'abbrev (mod_\d+) \(i : Int\) : Nat := (.*)$', otext, re.M))

    def eval_mod(name, i):
        e = re.sub(r'\(?\(?(sel_\d+) \(i\)\)?\)?', lambda mm: str(sel[mm.group(1)][i]), mods[name])
        return int(eval(e.replace('.toNat', '')))
    found = []
    omods = {k: int(v) for k, v in re.findall(r'abbrev (modulus_\d+) : Nat := (\d+)', otext)}
    for mm in re.finditer(r'let (w_\S+) : Mxx\.Primitives\.ExactMatrix (\d+|modulus_\d+) (\d+) 1 1 := '
                          r'\(MxxRuntime\.matrixPolynomial \[\(\((sel_\d+) \(\(Int\.ofNat i_0\)\)\)\)\] :', otext):
        kw, kq, kn, ks = mm.groups()
        kq = omods[kq] if kq in omods else int(kq)
        if kq != Q or int(kn) != N:
            continue
        start = otext.rfind('\nabbrev ', 0, mm.start())
        scope = re.match(r'\nabbrev (\S+)\.body ', otext[start:])
        if not scope:
            continue
        scope = scope.group(1)
        tm = re.search(rf'let (w_\S+) : [^\n]*:= MxxRuntime\.matrixMulScalarRight (w_\S+) {re.escape(kw)}\n',
                       otext[mm.start():mm.start() + 20000])
        if not tm:
            continue
        lw = tm.group(2)
        for cm in re.finditer(rf'abbrev {re.escape(scope)}\.constraints_\d+ (.*?) : Prop :=\n((?:  .*\n)*)', otext):
            rb = re.search(rf'MxxRuntime\.centeredRebaseRuns (w_\S+) {re.escape(lw)}\b', cm.group(2))
            if rb:
                ty = sig_params(cm.group(1)).get(rb.group(1), '')
                mt = re.match(r'Mxx\.Primitives\.ExactMatrix \((mod_\d+) \(\(Int\.ofNat i_0\)\)\)', ty)
                if mt:
                    found.append((scope, mt.group(1), ks))
    found = sorted(set(found))
    if len(found) == 1:
        scope, modn, seln = found[0]
        qs = [eval_mod(modn, i) for i in range(S)]
        ks_ = sel[seln][:S]
        check(len(ks_) == S, 'one CRT basis entry per instance')
        check(all(qs[i] == qs[(i // R) * R] and ks_[i] == ks_[(i // R) * R] for i in range(S)),
              'the CRT tables depend on i0 / R only')
        tables = dict(q=[qs[t * R] for t in range(K)], K=[ks_[t * R] for t in range(K)], scope=scope,
                      mod=modn, sel=seln)
        print(f'CRT-term role: {scope} ({modn}, {seln})', file=sys.stderr)
    else:
        print(f'CRT-term role not found uniquely in {online}: {found}; tables omitted', file=sys.stderr)

print(f'{R} components, {K} CRT terms, S = {S}, {m} columns, N = {N}, Q = {Q}', file=sys.stderr)

# ---------------------------------------------------------------------------------------------
# Emitting

c0 = comps[0]
MQ = f'ExactMatrix {Q} {N} 1 {m}'
Ldef = [
    'import GenDecode', f'import {NS}', '',
    '/-! The looped decode stage, from the generic lemmas of `GenDecode`',
    '(written by `generator/gen_decode_loops.py`). -/', '',
    'set_option maxRecDepth 65536', '',
    f'namespace {a.namespace}', '',
    'open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof GenDecode', '',
    f'/-! The stage\'s numbers: {R} components, {K} CRT terms per component, {S} family members,',
    f'{m} columns, ring dimension {N}, decoding modulus {Q}. -/', '',
    f'/-- The decode stage\'s inputs. -/\nabbrev Inputs := {unabbrev(inputs_ty)}', '',
    f'/-- The factor `r` of component `j`\'s expected row. -/',
    f'noncomputable def decodeR (inputs : Inputs) : Nat → ExactMatrix {Q} {N} 1 1']
for c in sorted(comps, key=lambda c: c['j']):
    Ldef.append(f'  | {c["j"]} => {c["r"]}')
Ldef.append('  | _ => 0')
Ldef += ['',
         '/-- The value whose CRT terms component `j` reads: its expected row plus the error row `E j`. -/',
         f'noncomputable def decodeValue (inputs : Inputs) (G : ExactMatrix {Qg} {N} 1 {m})',
         f'    (E : Nat → ErrorMatrix {N} 1 {m}) (j : Nat) : {MQ} :=',
         f'  decodeExpected (colBlock {m} {input_expr[c0["key"]]} ({m} * j))',
         f'    (fun _ _ ↦ polynomialOfCoefficients {input_expr[bits]}) {c0["delta"]} (decodeR inputs j)',
         f'    {input_expr[c0["coef"]]} {input_expr[c0["expo"]]} G + reduceMatrix {Q} {N} 1 {m} (E j)', '']


def child_pat(scope, keep):
    s = scopes[scope]

    def name_of(x):
        for pre, nm in keep:
            if x.startswith(pre):
                return nm
        check(TRIVIAL.match(x), f'{scope}: unrecognized atom {x}')
        return '-'
    return s.pattern(name_of)


adapters = []


def lead(k):
    return ' '.join(f'a{t}' for t in range(k))
for c in comps:
    sname = c['stride']['scope']
    pat = child_pat(sname, [('MxxRuntime.familyGetDynamic ', 'hx'), ('outputs = ', 'rfl')])
    check(scopes[sname].binders == 1 and not scopes[sname].witness, f'{sname}: one existential member')
    fam_pos = [i for i, x in enumerate(c['stride']['args']) if x == fam]
    check(len(fam_pos) == 1, f'{sname}: the family argument')
    names = ['fam' if i == fam_pos[0] else f'x{i + 1}' for i in range(len(c['stride']['args']))]
    adapters += [
        f'theorem stride_step_{sname} {{{lead(c["stride"]["lead"])}}} {{i : Nat}} {{cur next}} {{{" ".join(names)}}}',
        f'    (h : {NS}.{sname} {lead(c["stride"]["lead"])} i {nested(["cur"] + names)} next) :',
        f'    StrideStep fam {c["R"]} {c["j"]} i cur next := by',
        f'  obtain {pat.replace("⟨b0,", "⟨x,")} := h',
        '  exact ⟨x, hx, rfl⟩', '']
done = set()
for c in comps:
    cname = c['col']['scope']
    if cname in done:
        continue
    done.add(cname)
    info = col_info[cname]
    rname = info['round']
    rpat = child_pat(rname, [('MxxRuntime.familyGetDynamic ', 'hv'), ('outputs = ', 'rfl')])
    check(scopes[rname].binders == 1 and not scopes[rname].witness, f'{rname}: one existential value')
    rn = info['round_names']
    adapters += [
        f'theorem round_step_{rname} {{{lead(info["rlead"])}}} {{k : Nat}} {{c c\' : Int}} '
        f'{{{" ".join(rn[1:])}}}',
        f'    (h : {NS}.{rname} {lead(info["rlead"])} k {nested(rn)} c\') :',
        '    RoundStep vals half qc k c c\' := by',
        f'  obtain {rpat.replace("⟨b0,", "⟨v,")} := h',
        '  exact ⟨v, hv, rfl⟩', '']
    cpat = child_pat(cname, [('MxxRuntime.sliceMatrix ', 'hs'), ('MxxRuntime.polynomialValues ', 'hv'),
                             ('MxxIR.IterRuns ', 'hrun'), ('outputs = ', 'rfl')])
    check(scopes[cname].witness, f'{cname}: a witness scope')
    cn = info['col_names']
    adapters += [
        f'theorem column_step_{cname} {{{lead(c["col"]["lead"])}}} {{i : Nat}} {{cur next : Int}} '
        f'{{{" ".join(cn[1:])}}}',
        f'    (h : {NS}.{cname} {lead(c["col"]["lead"])} i {nested(cn)} next) :',
        '    ColumnStep D half qc i cur next := by',
        f'  obtain {cpat} := h',
        '  refine ⟨_, _, _, _, hs, hv, ?_, hrun⟩',
        f'  exact fun _ _ _ hb ↦ round_step_{rname} hb', '']

rpat = root.pattern(lambda x: used_atoms.get(x, '-'))
fam_expr = input_expr[fam]
thm = a.name
root_sig = sig_list(root.head)
root_lead = root_sig[:[nm for nm, _ in root_sig].index('inputs')]
binders, args = [], []
for k, (nm, ty) in enumerate(root_lead):
    ty = re.sub(r'(?<![\w.])Params\b', f'{NS}.Params', ty)
    nm = {'MxxRuntime.BackendContext': 'backend', 'MxxRuntime.HashModel': 'model'}.get(ty, nm)
    if nm == '_' or nm in args:
        nm = f'r{k}'
    binders.append(f'{{{nm} : {ty}}}')
    args.append(nm)
check('backend' in args and 'model' not in args, 'the root takes a backend and no hash model')
check(re.search(r'gadgetMatrixRuns backend ', gad[0]) and root_lead[args.index('backend')][0] == 'backend',
      'the gadget runs on the root backend')
root_binders = ' '.join(binders)
hyps = (f'    (h : {NS}.generatedRoot {" ".join(args)} inputs out)\n'
        f'    {{G : ExactMatrix {Qg} {N} 1 {m}}} (hG : gadgetMatrixRuns backend {gbase} {m} G)\n'
        f'    (E : Nat → ErrorMatrix {N} 1 {m})\n'
        f'    (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < ({Q} : Int))')
term = (f'    (hterm : ∀ i : Fin {S}, TermOf (qN (i / {R})) (KN (i / {R})) ({fam_expr} i)\n'
        f'      (decodeValue inputs G E (i % {R})))')
P = [f'/-- The decode stage outputs zero when every CRT term `T (t {R} + j)` it reads is the term of',
     f'`y_j + e\'_j` for the prime `qN t` and the basis element `KN t`, for the expected row `y_j` of',
     'component `j` and a small error row `e\'_j`. -/',
     f'theorem {thm}_gen (qN : Nat → Nat) (KN : Nat → Int) (hq : ∀ i : Fin {K}, 1 < qN i)',
     f'    (hK : ∀ i j : Fin {K}, KN i ≡ (if i = j then 1 else 0) [ZMOD (qN j : Int)])',
     f'    (hcop : ∀ i j : Fin {K}, i ≠ j → Nat.Coprime (qN i) (qN j)) (hprod : {Q} = ∏ i : Fin {K}, qN i)',
     f'    {root_binders}',
     '    {inputs : Inputs} {out : Int}',
     hyps, term + ' :', '    out = 0 := by',
     f'  obtain {rpat} := h',
     '  have hσ\' := sigma_of_values hσ',
     '  have hg := hreb.2',
     '  rw [gadgetMatrixRuns_unique hgad hG] at hg']
for c in sorted(comps, key=lambda c: c['j']):
    j = c['j']
    P += [f'  have z{j} := decode_component_zero (K := {K}) (Rn := {R}) (jn := {j}) (by decide) (by decide)',
          '    qN KN hq hK hcop hprod (by decide) (by decide)',
          f'    {fam_expr} (decodeValue inputs G E) hterm _ (E {j}) rfl (hE {j}) rfl rfl rfl rfl hinit{j}',
          f'    (fun _ _ _ hb ↦ stride_step_{c["stride"]["scope"]} hb) hs{j} ?hD{j} (by decide) rfl',
          f'    (fun _ _ _ hb ↦ column_step_{c["col"]["scope"]} hb) hc{j}',
          f'  case hD{j} =>',
          f'    rw [slice_row_gen hr{j}0 rfl (by decide), slice_row_gen hr{j}1 rfl (by decide), '
          f'slice_colBlock hA{j}, hσ\', hg]',
          '    rfl']
zs = ', '.join(f'z{c["j"]}' for c in sorted(comps, key=lambda c: c['j']))
P += [f'  rw [hout, {zs}]',
      '  try rfl', '']

T = []
if tables:
    vec = lambda xs: '[' + ', '.join(str(x) for x in xs) + ']'
    T = [f'/-- The CRT primes, read from the online stage\'s instance table `{tables["mod"]}` '
         f'(instance `t {R} + j`). -/',
         f'abbrev decode_qN (t : Nat) : Nat := {vec(tables["q"])}.getD t 0',
         f'/-- The CRT basis, read from the online stage\'s instance table `{tables["sel"]}`. -/',
         f'abbrev decode_KN (t : Nat) : Int := {vec(tables["K"])}.getD t 0', '',
         f'theorem decode_gt : ∀ i : Fin {K}, 1 < decode_qN i := by decide',
         f'theorem decode_basis : ∀ i j : Fin {K}, decode_KN i ≡ (if i = j then 1 else 0) '
         f'[ZMOD (decode_qN j : Int)] := by decide',
         f'theorem decode_coprime : ∀ i j : Fin {K}, i ≠ j → Nat.Coprime (decode_qN i) (decode_qN j) := by decide',
         f'theorem decode_prod : {Q} = ∏ i : Fin {K}, decode_qN i := by',
         '  simp only [Fin.prod_univ_succ, Fin.prod_univ_zero]', '  decide', '',
         f'/-- `{thm}_gen` for the online stage\'s CRT tables. -/',
         f'theorem {thm} {root_binders}',
         '    {inputs : Inputs} {out : Int}',
         hyps, term.replace('qN', 'decode_qN').replace('KN', 'decode_KN') + ' :', '    out = 0 :=',
         f'  {thm}_gen decode_qN decode_KN decode_gt decode_basis decode_coprime decode_prod h hG E hE hterm', '']

out = '\n'.join(Ldef + adapters + P + T + [f'end {a.namespace}', ''])
open(out_path, 'w').write(out)
print(f'wrote {out_path}', file=sys.stderr)
