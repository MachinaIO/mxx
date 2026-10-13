#!/usr/bin/env python3
"""Computation bundles of encoding-level scopes (helper of gen_adapters.py, WP C).

A *bundle* of a scope is its relation with everything instance-specific abstracted:

* matrix dimensions become `Nat` parameters, except where the typing of the runtime operations
  forces a relation (`concatRows`/`concatColumns` add, `tensorRuns` multiplies, `matrixMul`,
  `matrixAdd`, ... unify), which is kept as an expression (`d0 + d0`); row/column/family sizes
  `<= 3` (the row structure of the encodings) stay literal;
* moduli become `Nat` parameters (one per class of matrices forced to share the modulus), the
  ring dimension the parameter `n`;
* numerals `> 3`, constant lists, rationals, `sel_k (...)`, `mod_k (...)`, `modulus_k` and loop
  counts become parameters, one per occurrence (never identified by value);
* `let`-bound wires are inlined, input wires become `inputs` projections, existential and
  witness wires become the bundle's binders `v0, v1, ...`; comparisons that mention no wire
  (instance bounds, `0 <= 0`) are dropped;
* a sequential child loop's body relation becomes the parameter `child`; a parallel child loop
  with no inputs and one output family (the value table of a LUT key table) becomes the parameter
  `table` (`table i o`: member `i` of the family is `o`).

`make(scope, role)` returns a `Bundle`: the canonical `Comp` text (equal for all scopes of a
role variant), this scope's parameter values and the adapter data (binder order, kept atoms).
"""
import re

import xmodel

PRIMS = 'Mxx.Primitives.ExactMatrix'
WIRE_RE = re.compile(r'\bw_\d+(?:_\d+)?(?:_concat_\d+)?\b')


class Fail(Exception):
    pass


# --------------------------------------------------------------------------- types

def parse_type(t):
    t = xmodel.strip_parens(t.strip())
    items = xmodel.split_top(t, ' × ')
    if len(items) > 1:
        return ('prod', [parse_type(x) for x in items if xmodel.strip_parens(x) != 'Unit'])
    if t == 'Unit':
        return ('unit',)
    parts = xmodel.split_top(t, ' → ')
    if len(parts) > 1:
        dep = re.match(r'^\((\w+) : (Fin \S+)\)$', parts[0].strip())
        if dep:
            if re.search(rf'\b{dep.group(1)}\b', ' → '.join(parts[1:])):
                raise Fail('dependent family type')
            parts[0] = dep.group(2)
        m = re.match(r'^Fin (\S+)$', parts[0].strip())
        if not m:
            raise Fail(f'function type {t[:80]}')
        return ('fam', m.group(1), parse_type(' → '.join(parts[1:])))
    if t.startswith(PRIMS + ' ') or t.startswith('ExactMatrix '):
        rest = t.split(' ', 1)[1]
        a = [p for p in xmodel.split_top(rest, ' ') if p]
        if len(a) != 4:
            raise Fail(f'matrix type {t[:80]}')
        return ('mat', a)
    return ('atom', t)


class Dims:
    """Union-find over dimension slots with sum/product definitions."""

    def __init__(self):
        self.parent, self.val, self.defn, self.kind = {}, {}, {}, {}

    def add(self, key, value, kind):
        self.parent[key] = key
        self.val[key] = value
        self.kind[key] = kind
        return key

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.val[a] != self.val[b]:
            raise Fail(f'dimension clash {self.val[a]} vs {self.val[b]}')
        self.parent[b] = a
        if b in self.defn and a not in self.defn:
            self.defn[a] = self.defn.pop(b)

    def define(self, k, op, x, y):
        k = self.find(k)
        if k not in self.defn:
            self.defn[k] = (op, x, y)


class Builder:
    def __init__(self):
        self.d = Dims()
        self.wt = {}
        self.slot = 0
        self.state = None

    def tree(self, t, owner):
        return self._slots(parse_type(t), owner)

    def _slots(self, p, owner):
        if p[0] == 'mat':
            ks = []
            for nm, v in zip('qnrc', p[1]):
                self.slot += 1
                ks.append(self.d.add(f'{owner}#{nm}{self.slot}', v, nm))
            return ('mat', ks)
        if p[0] == 'fam':
            self.slot += 1
            return ('fam', self.d.add(f'{owner}#F{self.slot}', p[1], 'F'), self._slots(p[2], owner))
        if p[0] == 'prod':
            return ('prod', [self._slots(x, owner) for x in p[1]])
        return p

    def unify(self, a, b):
        if a[0] != b[0]:
            raise Fail(f'type shape {a[0]} vs {b[0]}')
        if a[0] == 'mat':
            for x, y in zip(a[1], b[1]):
                self.d.union(x, y)
        elif a[0] == 'fam':
            self.d.union(a[1], b[1])
            self.unify(a[2], b[2])
        elif a[0] == 'prod':
            if len(a[1]) != len(b[1]):
                raise Fail('product arity')
            for x, y in zip(a[1], b[1]):
                self.unify(x, y)
        elif a[0] == 'atom' and a[1] != b[1]:
            raise Fail(f'atom types {a[1]} {b[1]}')

    def T(self, w):
        if w not in self.wt:
            raise Fail(f'untyped wire {w}')
        return self.wt[w]

    def term_type(self, t):
        t = xmodel.strip_parens(t)
        if WIRE_RE.fullmatch(t):
            return self.T(t)
        m = re.fullmatch(r'current((?:\.2)*)\.1', t)
        if m and self.state is not None:
            k = len(m.group(1)) // 2
            return self.state[1][k] if self.state[0] == 'prod' else self.state
        if t in ('current', 'next') and self.state is not None:
            return self.state
        raise Fail(f'cannot type {t[:60]}')

    def mat(self, t):
        x = self.term_type(t)
        if x[0] != 'mat':
            raise Fail(f'not a matrix {t[:60]}')
        return x[1]

    def eq_qn(self, a, b):
        self.d.union(a[0], b[0])
        self.d.union(a[1], b[1])

    def eq_all(self, a, b, start=0):
        for x, y in list(zip(a, b))[start:]:
            self.d.union(x, y)

    def constrain(self, head, args, res=None):
        h = head.split('.')[-1]
        M = self.mat
        if h in ('matrixAdd', 'matrixSub'):
            self.eq_all(M(args[0]), M(args[1]))
            if res:
                self.eq_all(M(args[0]), res)
        elif h == 'matrixNeg':
            if res:
                self.eq_all(M(args[0]), res)
        elif h == 'matrixMul':
            a, b = M(args[0]), M(args[1])
            self.eq_qn(a, b)
            self.d.union(a[3], b[2])
            if res:
                self.eq_qn(a, res)
                self.d.union(a[2], res[2])
                self.d.union(b[3], res[3])
        elif h == 'matrixMulScalarLeft':
            if res:
                self.eq_all(M(args[1]), res)
                self.eq_qn(M(args[0]), res)
        elif h == 'matrixMulScalarRight':
            if res:
                self.eq_all(M(args[0]), res)
                self.eq_qn(M(args[1]), res)
        elif h == 'multiplyMonomial':
            if res:
                self.eq_all(M(args[0]), res)
        elif h == 'concatRows':
            a, b, c = M(args[0]), M(args[1]), M(args[2])
            self.eq_qn(a, b)
            self.eq_qn(a, c)
            self.d.union(a[3], b[3])
            self.d.union(a[3], c[3])
            self.d.define(c[2], '+', a[2], b[2])
        elif h == 'concatColumns':
            a, b, c = M(args[0]), M(args[1]), M(args[2])
            self.eq_qn(a, b)
            self.eq_qn(a, c)
            self.d.union(a[2], b[2])
            self.d.union(a[2], c[2])
            self.d.define(c[3], '+', a[3], b[3])
        elif h == 'tensorRuns':
            a, b, c = M(args[0]), M(args[1]), M(args[2])
            self.eq_qn(a, b)
            self.eq_qn(a, c)
            self.d.define(c[2], '*', a[2], b[2])
            self.d.define(c[3], '*', a[3], b[3])
        elif h == 'sliceMatrix':
            self.eq_qn(M(args[0]), M(args[5]))
        elif h == 'gadgetDecomposeRuns':
            a, b = M(args[3]), M(args[4])
            self.eq_qn(a, b)
            self.d.union(a[3], b[3])
        elif h == 'compactCenteredRebaseRuns':
            self.eq_all(M(args[1]), M(args[2]), 1)
        elif h in ('centeredRebaseRuns', 'modulusReduceRuns', 'modulusSwitchRuns'):
            self.eq_all(M(args[0]), M(args[1]), 1)
        elif h == 'blockModSwitchRuns':
            self.eq_all(M(args[2]), M(args[3]), 1)
        elif h == 'ringAutomorphismRuns':
            self.eq_all(M(args[1]), M(args[2]))
        elif h in ('familyGetDynamic', 'familyGetStatic'):
            f = self.term_type(args[0])
            self.unify(f[2], self.term_type(args[2]))
        elif h in ('hashSample', 'gaussianSample', 'gadgetMatrixRuns', 'uniformIntervalSample',
                   'liftInteger', 'matrixPolynomial', 'allCoefficients', 'rotationPolynomial', 'unitRow', 'selectInt',
                   'trapdoorSample', 'preimageRunsDispatched', 'polynomialFromValues',
                   'polynomialValues', 'select'):
            pass
        else:
            raise Fail(f'no typing rule for {h}')


# --------------------------------------------------------------------------- rendering

class Render:
    def __init__(self, B):
        self.B = B
        self.dimnames = {}
        self.params = []       # (name, type, value, 'dim'|'const')
        self.pcount = 0

    def dim(self, k):
        d = self.B.d
        r = d.find(k)
        if r in d.defn:
            op, x, y = d.defn[r]
            return f'({self.dim(x)} {op} {self.dim(y)})'
        v, kind = d.val[r], d.kind[r]
        if re.fullmatch(r'\d+', v) and ((kind == 'r' and int(v) <= 3) or (kind == 'c' and int(v) == 1)):
            return v
        if r not in self.dimnames:
            base = {'n': 'n', 'q': 'q'}.get(kind, 'd')
            idx = sum(1 for x in self.dimnames.values() if x.startswith(base))
            nm = 'n' if base == 'n' and idx == 0 else f'{base}{idx}'
            self.dimnames[r] = nm
            self.params.append((nm, 'Nat', v, 'dim'))
        return self.dimnames[r]

    def typ(self, tr):
        if tr[0] == 'mat':
            q, n, r, c = (self.dim(k) for k in tr[1])
            return f'ExactMatrix {q} {n} {r} {c}'
        if tr[0] == 'fam':
            return f'(Fin {self.dim(tr[1])} → {self.typ(tr[2])})'
        if tr[0] == 'prod':
            return ' × '.join(self.typ(x) for x in tr[1]) + ' × Unit'
        if tr[0] == 'unit':
            return 'Unit'
        return tr[1].replace('MxxRuntime.', '')

    def param(self, value, ty='_'):
        nm = f'p{self.pcount}'
        self.pcount += 1
        self.params.append((nm, ty, value, 'const'))
        return nm


SMALL = re.compile(r'^-?[0-3]$')


def abstract_consts(text, R, small=True):
    """Replace the instance constants of `text` by bundle parameters (`small=False`: also the
    numerals `<= 3`)."""
    out = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        prev = text[i - 1] if i else ' '
        if ch == '[':
            j = xmodel.matching(text, i)
            inner = text[i:j + 1]
            if not WIRE_RE.search(inner) and 'current' not in inner and 'inputs' not in inner:
                out.append(R.param(inner))
            else:
                out.append('[' + abstract_consts(text[i + 1:j], R, small) + ']')
            i = j + 1
            continue
        if ch == '(':
            j = xmodel.matching(text, i)
            inner = text[i + 1:j].strip()
            if re.fullmatch(r'[\d\s+\-*/()]+', inner) and re.search(r'\d', inner):
                bare = inner.replace('(', '').replace(')', '').strip()
                ctx = ''.join(out)[-60:]
                index_pos = re.search(r'(selectInt |familyGetStatic \S+ |rotationPolynomial )$', ctx)
                if SMALL.match(bare) and small and not index_pos:
                    out.append(text[i:j + 1])
                else:
                    out.append('(' + R.param(text[i:j + 1]) + ')')
                i = j + 1
                continue
        m = re.match(r'\(MxxIR\.selectInt \(\(Int\.ofNat i_0\)\) \[', text[i:])
        if m and ch == '(':
            j = xmodel.matching(text, i)
            out.append(R.param(text[i:j + 1]))
            i = j + 1
            continue
        m = re.match(r'(sel_\d+|mod_\d+) ', text[i:])
        if m and not (prev.isalnum() or prev in '_.'):
            k = i + m.end()
            if k < n and text[k] == '(':
                j = xmodel.matching(text, k)
                out.append(R.param('(' + text[i:j + 1] + ')'))
                i = j + 1
                continue
        m = re.match(r'modulus_\d+\b', text[i:])
        if m and not (prev.isalnum() or prev in '_.'):
            out.append(R.param(m.group(0)))
            i += m.end()
            continue
        m = re.match(r'\d+', text[i:])
        if m and not (prev.isalnum() or prev in '_.'):
            num = m.group(0)
            out.append(R.param(num) if (int(num) > 3 or not small) else num)
            i += m.end()
            continue
        m = re.match(r"[A-Za-z_][A-Za-z_0-9.']*", text[i:])
        if m:
            out.append(m.group(0))
            i += m.end()
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def abstract_hash(t, R):
    """`hashSample model tag comps key out`: the tag prefix and every numeral of the components
    except the trailing component index become parameters; `.bytes` literals stay."""
    h, a = xmodel.app_split(t)
    if not h.endswith('hashSample') or len(a) != 5:
        return abstract_consts(t, R)
    model, tag, comps, key, out = a
    inner = xmodel.strip_parens(comps)
    items = [x.strip() for x in xmodel.split_top(inner[1:-1], ',')]
    new = []
    for j, it in enumerate(items):
        if it.startswith('.bytes') or j == len(items) - 1:
            new.append(it)
            continue
        mm = re.match(r'^(\.\w+) (.*)$', it)
        if mm and not WIRE_RE.search(mm.group(2)) and 'Int.ofNat' not in mm.group(2):
            new.append(f'{mm.group(1)} ({R.param(mm.group(2))})')
        else:
            new.append(abstract_consts(it, R))
    return f'{h} {model} {R.param(tag)} ([{", ".join(new)}]) {key} {out}'


class Bundle:
    pass


def make(s, role):
    B = Builder()
    types = {}
    for k, (binders, _) in s.frag_src.items():
        for nm, t in binders:
            if nm.startswith('w_'):
                types[nm] = t
    for w in s.wires.values():
        if w.type and w.name not in types:
            types[w.name] = w.type
    for nm, t in s.witness:
        types.setdefault(nm, t)
    for nm, t in types.items():
        B.wt[nm] = B.tree(t, nm)
    lets = {w.name: w for w in s.wires.values() if w.src in ('let', 'input')}
    for w in lets.values():
        if w.src == 'let' and w.term and w.name in B.wt and B.wt[w.name][0] == 'mat':
            h, a = xmodel.app_split(w.term)
            if h.startswith('MxxRuntime.'):
                B.constrain(h, a, B.wt[w.name][1])
    atoms = []
    for k in sorted(s.frag_src):
        atoms += [t.strip() for t in xmodel.split_conj(s.frag_src[k][1])]
    for t in atoms:
        lp = xmodel.parse_loop(t)
        if lp is not None:
            if lp.kind == 'par':
                continue
            if lp.kind != 'seq':
                raise Fail('gather loop inside an encoding-level scope')
            m = re.search(r'\(current next : (.*?)\) => ', t)
            B.state = B.tree(m.group(1), '$state')
            for tup in (lp.init, lp.final):
                items = xmodel.tuple_items(tup)
                if B.state[0] == 'prod':
                    for j, x in enumerate(items):
                        B.unify(B.state[1][j], B.term_type(x))
                else:
                    B.unify(B.state, B.term_type(items[0]))
            continue
        h, a = xmodel.app_split(t)
        if h.startswith('MxxRuntime.'):
            B.constrain(h, a)
    R = Render(B)
    ex_order = re.findall(r'∃ \((w_\d+(?:_\d+)?(?:_concat_\d+)?) : ', s.def_src)
    wit = [nm for nm, _ in s.witness]
    binders = wit + [w for w in ex_order if w not in wit]
    vname = {w: f'v{k}' for k, w in enumerate(binders)}
    idx_orig = [nm for nm, t in s.sig if t == 'Nat']
    # index binders: the instance index `i0`, the scope's own (last) index `i`, others `j<k>`;
    # only binders occurring in the scope's atoms or lets are kept
    idx_new = []
    for k, o in enumerate(idx_orig):
        idx_new.append('i0' if k == 0 else ('i' if k == len(idx_orig) - 1 else f'j{k}'))
    used_text = ' '.join(atoms) + ' ' + ' '.join((w.term or '') for w in s.wires.values())

    def ren_idx(t):
        for o, nn in zip(idx_orig, idx_new):
            if o != '_':
                t = re.sub(rf'\b{o}\b', nn, t)
        return t
    idx_used = [nn for o, nn in zip(idx_orig, idx_new) if o != '_' and re.search(rf'\b{o}\b', used_text)]

    def let_term(w):
        term = lets[w].term
        mm = re.match(r'^\((.*) : ' + re.escape(PRIMS) + r' .*\)$', term)
        if mm:
            return f'({abstract_consts(mm.group(1), R)} : {R.typ(B.wt[w])})'
        if w in B.wt and B.wt[w][0] == 'mat':
            return f'({abstract_consts(term, R)} : {R.typ(B.wt[w])})'
        # the LUT key table's monomial strides are instance constants even when small
        return abstract_consts(term, R, small=not role.startswith('lut.key-table'))

    def inl(t, depth=0):
        if depth > 80:
            raise Fail('inline depth')

        def rep(m):
            w = m.group(0)
            if w in vname:
                return vname[w]
            lw = lets.get(w)
            if lw is None:
                raise Fail(f'unknown wire {w}')
            if lw.src == 'input':
                return lw.term
            return '(' + inl(let_term(w), depth + 1) + ')'
        return WIRE_RE.sub(rep, t)

    kept, body_atoms, child, child_ty, table = [], [], None, None, None
    out_k = None
    for k, t in enumerate(atoms):
        lp = xmodel.parse_loop(t)
        if lp is not None and lp.kind == 'par':
            # `∀ i : Fin N, Child <mid> i () (w i)`: an input-free table family
            body = xmodel.strip_parens(t)
            body = body[re.match(r'^∀ (\w+) : Fin (\d+), ', body).end():]
            h, a = xmodel.app_split(body)
            out = re.fullmatch(r'\(?(w_\d+(?:_\d+)?) ' + lp.var + r'\)?', a[-1].strip())
            if table is not None or xmodel.strip_parens(a[-2]) not in ('()', '') or not out:
                raise Fail('parallel child loop of unknown form')
            if a[-3].strip() != lp.var:
                raise Fail('parallel child loop index position')
            table = (lp.child, ' '.join(a[:-3]))
            ft = B.T(out.group(1))
            if ft[0] != 'fam':
                raise Fail('table output is not a family')
            body_atoms.append(f'(∀ {lp.var} : Fin {R.dim(ft[1])}, table {lp.var} ({inl(out.group(1))} {lp.var}))')
            kept.append(k)
            continue
        if lp is not None:
            m = re.search(r'\(current next : (.*?)\) => (\S+) (.*?) (\(.*\)) next\) \(Int\.toNat \((.*?)\)\) ', t)
            if not m:
                raise Fail('loop atom form')
            child = (lp.child, m.group(3))
            st = R.typ(B.state)
            cargs = xmodel.tuple_items(m.group(4))
            ctys = [R.typ(B.term_type(x)) for x in cargs]
            child_ty = f'Nat → {" × ".join(ctys + ["Unit"])} → {st} → Prop'
            init = '(' + inl(abstract_consts(lp.init, R)) + ')'
            final = '(' + inl(abstract_consts(lp.final, R)) + ')'
            args = inl(abstract_consts(m.group(4), R))
            cnt = R.param('(' + m.group(5) + ')')
            body_atoms.append(f'MxxIR.IterRuns (fun (i : Nat) (current next : {st}) => child i {args} next) '
                              f'(Int.toNat {cnt}) {init} {final}')
            kept.append(k)
            continue
        if t.startswith('outputs = '):
            out_k = k
        elif not WIRE_RE.search(t):
            continue
        if xmodel.classify(t, None)[0] == 'cmp':
            body_atoms.append(inl(abstract_consts(t, R, small=False)))
            kept.append(k)
            continue
        body_atoms.append(inl(abstract_hash(t, R)))
        kept.append(k)
    in_wires = [w for w in s.wires.values() if w.src == 'input']
    tin = s.input_type
    if not tin or xmodel.strip_parens(tin) == 'Unit':
        tin_r = 'Unit'
    elif ' × ' in tin:
        tin_r = ' × '.join(R.typ(B.T(w.name)) for w in in_wires) + ' × Unit'
    else:
        tin_r = R.typ(B.T(in_wires[0].name))
    if out_k is None:
        raise Fail('no output equation')
    ocomps = xmodel.tuple_items(atoms[out_k][len('outputs = '):])
    if s.output_type and ' × ' in s.output_type:
        tout_r = ' × '.join(R.typ(B.term_type(x)) for x in ocomps) + ' × Unit'
    else:
        tout_r = R.typ(B.term_type(atoms[out_k][len('outputs = '):]))
    exs = ' '.join(f'({vname[w]} : {R.typ(B.T(w))})' for w in binders)
    body = ' ∧\n    '.join(body_atoms)
    b = Bundle()
    b.role = role
    b.dims = [p for p in R.params if p[3] == 'dim']
    b.consts = [p for p in R.params if p[3] == 'const']
    b.binders, b.kept, b.atoms, b.child, b.idx_orig = binders, kept, atoms, child, idx_orig
    b.table = table
    _fin = ren_idx(body) + ' ' + ren_idx(child_ty or '')
    idx_used = [nn for nn in idx_used if re.search(rf'\b{nn}\b', _fin)]
    b.idx_used = [o for o, nn in zip(idx_orig, idx_new) if nn in idx_used]
    head = ' '.join(f'({nm} : Nat)' for nm, _, _, _ in b.dims)
    head += ' ' + ' '.join(f'({nm} : {ty})' for nm, ty, _, _ in b.consts)
    head += ' (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel)'
    if idx_used:
        head += ' ' + ' '.join(f'({x} : Nat)' for x in idx_used)
    if child_ty:
        head += f' (child : {ren_idx(child_ty)})'
    if table:
        head += ' (table : Nat → Int → Prop)'
    head += f' (inputs : {tin_r}) (outputs : {tout_r})'
    b.head = head
    body = ren_idx((f'∃ {exs},\n    ' if exs else '') + body)
    prev = None
    while prev != body:
        prev = body
        body = re.sub(r'\(\((p\d+)\)\)', r'(\1)', body)
    b.body = body
    return b
