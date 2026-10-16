"""Deterministic cross-stage agreement proofs for gen_s0o.py.

Given two exported stage modules whose key computations are textually identical
child scopes (the online stage and its keys stage), this module proves that
selected root wires agree.  Agreement is shown wire by wire from the producing
Facts: uniqueness of every deterministic runtime predicate, unfolding of
definitional lets, determinism of nested parallel/sequential/gather loops via
per-scope agreement lemmas, and claim-level equality of the root inputs.  Rows
1 and 2 of a stacked `[c; A0; A1]` product read only key rows, so a slice of
such a product never demands the ciphertext row.  Nothing about ciphertexts,
plaintexts or noise is assumed or derived.

Only the dataflow actually demanded is traversed; scope pairs are checked to be
textually identical up to scope numbering before a lemma is emitted for them.
Tuple positions are projection paths, so balanced and right-nested tuples are
both supported.
"""
import re
from pathlib import Path

import dio2_meta as meta
import xmodel
from xmodel import app_split, split_top, strip_parens, matching
from xmodules import module_sources

WIRE = re.compile(r'\bw_\d+(?:_\d+)?(?:_gathered|_concat_\d+)?\b')
CHILD = re.compile(r'\b((?:parallel|sequential|scope)_[A-Za-z_0-9]+) ')

UNIQUE = {
    'modulusReduceRuns': 'modulusReduceRuns_unique',
    'centeredRebaseRuns': 'centeredRebaseRuns_unique',
    'compactCenteredRebaseRuns': 'compactCenteredRebaseRuns_unique',
    'gadgetDecomposeRuns': 'gadgetDecomposeRuns_unique',
    'gadgetMatrixRuns': 'gadgetMatrixRuns_unique',
    'concatRows': 'concatRows_unique',
    'concatColumns': 'concatColumns_unique',
    'sliceMatrix': 'sliceMatrix_unique',
    'ringAutomorphismRuns': 'ringAutomorphismRuns_unique',
    'familyGetDynamic': 'familyGetDynamic_unique',
    'familyGetStatic': 'familyGetDynamic_unique',
    'select': 'Step0Agree.select_unique',
    'familyPack': 'Step0Agree.familyPack_unique',
    'tensorRuns': 'Step0Agree.tensorRuns_unique',
    'blockDiagonalRuns': 'Step0Agree.blockDiagonalRuns_unique',
}


class AgreeError(meta.MetadataError):
    pass


def req(cond, message):
    if not cond:
        raise AgreeError(message)


def shape(text):
    t = re.sub(r'generatedRoot_\d+', 'gR', text)
    t = re.sub(r'\bscope_\d+', 'sc', t)
    return WIRE.sub('W', t)


# --------------------------------------------------------------------------- tuple trees

def parse_tuple(text):
    """A tuple term as a tree: a leaf string, or a pair (left, right)."""
    s = text.strip()
    if len(split_top(s, ',')) > 1:
        s = '(' + s + ')'
    if s.startswith('(') and matching(s, 0) == len(s) - 1:
        inner = s[1:-1].strip()
        if inner == '':
            return '()'
        parts = [p.strip() for p in split_top(inner, ',')]
        if len(parts) == 1:
            return parse_tuple(inner) if inner.startswith('(') else inner
        tree = parse_tuple(parts[-1])
        for p in reversed(parts[:-1]):
            tree = (parse_tuple(p), tree)
        return tree
    return s


def navigate(tree, path):
    for comp in re.findall(r'\.(\d)', path):
        req(isinstance(tree, tuple), f'cannot project {path}')
        tree = tree[0] if comp == '1' else tree[1]
    return tree


def leaves(tree, path=''):
    if isinstance(tree, tuple):
        return leaves(tree[0], path + '.1') + leaves(tree[1], path + '.2')
    if tree == '()':
        return []
    return [(path, tree)]


def call_args(atom):
    """(child scope, argument tree, output tree) of a loop atom."""
    text = strip_parens(atom.text)
    if atom.loop.kind == 'seq':
        fun = strip_parens(app_split(text)[1][0])
        body = fun.split('=> ', 1)[1]
    else:
        body = re.sub(r'^∀ \w+ : Fin \d+, ', '', text)
        if atom.loop.kind == 'gather':
            body = [c.strip() for c in split_top(strip_exists(body), ' ∧ ')][-1]
    head, args = app_split(body)
    req(CHILD.match(head + ' '), f'no child call in {text[:80]}')
    return head, parse_tuple(args[-2]), parse_tuple(args[-1])


def strip_exists(body):
    while body.startswith('∃ '):
        mm = re.match(r'^∃ (\w+) : ', body)
        rest = body[mm.end():]
        ty = split_top(rest, ', ')[0]
        body = rest[len(ty) + 2:]
    return body


def gather_conj(atom):
    body = re.sub(r'^∀ \w+ : Fin \d+, ', '', strip_parens(atom.text))
    binders = re.findall(r'∃ (w_\d+_\d+_gathered) : ', body)
    return binders, [c.strip() for c in split_top(strip_exists(body), ' ∧ ')]


class ScopeInfo:
    def __init__(self, module, scope, outs_text):
        self.module = module
        self.s = scope
        self.name = scope.name
        self.record = scope.kind == 'record'
        self.prod = {}
        for a in scope.atoms:
            if a.loop:
                if a.loop.kind == 'seq':
                    out_tree = parse_tuple(a.loop.final)
                else:
                    out_tree = call_args(a)[2]
                for path, leaf in leaves(out_tree):
                    b = strip_parens(leaf).split()[0]
                    if WIRE.fullmatch(b):
                        self.prod[b] = ('loop', a, path)
            elif a.kind == 'out':
                continue
            elif a.kind == 'sha256BitColumns':
                self.prod.setdefault(strip_parens(a.args[-2]), ('atom', a))
                self.prod.setdefault(strip_parens(a.args[-1]), ('atom', a))
            elif a.args and WIRE.fullmatch(strip_parens(a.args[-1])):
                self.prod.setdefault(strip_parens(a.args[-1]), ('atom', a))
        self.out_atom = next((a for a in scope.atoms if a.kind == 'out'), None)
        self.exists = re.findall(r'∃ \((w_\d+(?:_\d+)?(?:_concat_\d+)?) : ', scope.def_src or '')
        self.frags = sorted({a.frag for a in scope.atoms})
        self.conj = {k: [a for a in scope.atoms if a.frag == k] for k in self.frags}
        self.outs_tree = None
        if outs_text is not None:
            tree = parse_tuple(outs_text)
            self.outs_tree = self.strip_refs(tree)

    def strip_refs(self, tree):
        if isinstance(tree, tuple):
            return (self.strip_refs(tree[0]), self.strip_refs(tree[1]))
        if tree == '()':
            return tree
        m = re.fullmatch(r'\(?' + re.escape(self.name) + r'\.(w_\S+) ctx\)?', tree.strip())
        req(m is not None, f'{self.name}: unexpected outs item {tree}')
        return m.group(1)

    def node(self, w):
        if w in self.prod:
            return self.prod[w]
        wi = self.s.wires.get(w)
        req(wi is not None, f'{self.name}: unknown wire {w}')
        if wi.src == 'input':
            return ('input', wi.term[len('inputs'):])
        if wi.src == 'let':
            term = wi.term.split('\n\n')[0]
            term = re.sub(r'\n\s*end\s+\S+\s*$', '', term).strip()
            return ('let', term)
        raise AgreeError(f'{self.name}: no producer for {w} ({wi.src})')

    def outputs(self):
        """(path, wire) of every output leaf."""
        if self.record:
            req(self.outs_tree is not None, f'{self.name}: no outs definition')
            tree = self.outs_tree
        else:
            req(self.out_atom is not None, f'{self.name}: no out atom')
            tree = parse_tuple(self.out_atom.args[0])
        return [(p, strip_parens(x)) for p, x in leaves(tree)]


class Lemma:
    """One agreement theorem for a pair of textually identical scopes."""

    def __init__(self, gen, A, B, outs, name, mode=None):
        self.gen = gen
        self.A, self.B = A, B
        self.outs = outs          # demanded output paths
        self.name = name
        self.lines = []
        self.blocks = []          # (name, statement, lines)
        self.mark = 0
        self.memo = {}
        self.inputs = {}          # input path -> hypothesis name
        self.counter = 0
        self.mode = mode or ('record' if A.record else 'chain')
        self.gathered = {}

    def close(self, name, wa, wb):
        stmt = f'{self.term("A", wa)} = {self.term("B", wb)}'
        self.blocks.append((name, stmt, self.lines[self.mark:]))
        self.mark = len(self.lines)

    def fresh(self, base='e'):
        self.counter += 1
        return f'{base}{self.counter}'

    def term(self, side, w):
        S = self.A if side == 'A' else self.B
        if self.mode in ('record', 'root'):
            ctx = 'co' if side == 'A' else 'ck'
            return f'({S.module}.{S.name}.{w} {ctx})'
        return w if side == 'A' else w + "'"

    def hyp(self, side, atom):
        if self.mode in ('record', 'root'):
            F = 'Fo' if side == 'A' else 'Fk'
            return f'{F}.c{atom.frag}.{atom.field}'
        S = self.A if side == 'A' else self.B
        group = S.conj[atom.frag]
        j = next(k for k, a in enumerate(group) if a is atom)
        return f'{"cA" if side == "A" else "cB"}{atom.frag}{and_proj(j, len(group))}'

    def eq(self, wa, wb):
        key = (wa, wb)
        if key in self.memo:
            return self.memo[key]
        na, nb = self.A.node(wa), self.B.node(wb)
        req(na[0] == nb[0], f'{self.A.name}.{wa} {na[0]} vs {self.B.name}.{wb} {nb[0]}')
        kind = na[0]
        if kind == 'input':
            result = [self.input_eq(na[1], nb[1], wa, wb)]
        elif kind == 'let':
            ta, tb = na[1], nb[1]
            req(shape(ta) == shape(tb), f'let shape {wa}={ta} vs {wb}={tb}')
            subs = self.collect(WIRE.findall(ta), WIRE.findall(tb))
            if self.mode in ('record', 'root'):
                name = self.fresh()
                lhs, rhs = self.term('A', wa), self.term('B', wb)
                if subs:
                    body = (f'by unfold {self.A.module}.{self.A.name}.{wa} '
                            f'{self.B.module}.{self.B.name}.{wb}; rw [{", ".join(subs)}]')
                else:
                    body = 'rfl'
                self.lines.append(f'  have {name} : {lhs} = {rhs} := {body}')
                self.close(name, wa, wb)
                result = [name]
            else:
                result = subs
        elif kind == 'atom':
            result = [self.atom_eq(wa, wb, na[1], nb[1])]
        elif kind == 'loop':
            result = [self.loop_eq(wa, wb, na, nb)]
        else:
            raise AgreeError(f'unsupported node {kind}')
        self.memo[key] = result
        return result

    def collect(self, xs, ys):
        req(len(xs) == len(ys), 'wire count mismatch')
        subs = []
        for x, y in zip(xs, ys):
            for s in self.eq(x, y):
                if s not in subs:
                    subs.append(s)
        return subs

    def input_eq(self, pa, pb, wa, wb):
        req(pa == pb, f'{self.A.name}: input paths differ {pa} {pb}')
        if pa not in self.inputs:
            self.inputs[pa] = f'h{len(self.inputs)}'
        name = self.inputs[pa]
        if self.mode == 'record':
            iname = f'i{name}'
            if not any(l.startswith(f'  have {iname} ') for l in self.lines):
                self.lines.append(f'  have {iname} : {self.term("A", wa)} = {self.term("B", wb)} := {name}')
            return iname
        return name

    # ------------------------------------------------------------------ atoms
    def stack_parts(self, S, X):
        nx = S.node(X)
        if nx[0] != 'let':
            return None
        h, args = app_split(nx[1])
        if h != 'MxxRuntime.matrixMul' or len(args) != 2:
            return None
        both, d = strip_parens(args[0]), strip_parens(args[1])
        if not (WIRE.fullmatch(both) and WIRE.fullmatch(d)):
            return None
        nb = S.node(both)
        if nb[0] != 'atom' or nb[1].kind != 'concatColumns':
            return None
        rows, atoms = [], []
        for side in (strip_parens(nb[1].args[0]), strip_parens(nb[1].args[1])):
            ns = S.node(side)
            if ns[0] != 'atom' or ns[1].kind != 'concatRows':
                return None
            n01 = S.node(strip_parens(ns[1].args[0]))
            if n01[0] != 'atom' or n01[1].kind != 'concatRows':
                return None
            atoms += [n01[1], ns[1]]
            rows.append([strip_parens(n01[1].args[0]), strip_parens(n01[1].args[1]),
                         strip_parens(ns[1].args[1])])
        atoms.append(nb[1])
        return d, atoms, rows

    def one(self, names):
        if len(names) == 1:
            return names[0]
        return f'(by rw [{", ".join(names)}])' if names else 'rfl'

    def stack_slice_eq(self, wa, wb, aa, ab):
        if aa.kind != 'sliceMatrix':
            return None
        a1, a2, a3 = (strip_parens(x) for x in aa.args[1:4])
        if not (a1.isdigit() and a2 == str(int(a1) + 1) and a3 == '0' and int(a1) <= 2):
            return None
        r = int(a1)
        pa = self.stack_parts(self.A, strip_parens(aa.args[0]))
        pb = self.stack_parts(self.B, strip_parens(ab.args[0]))
        if pa is None or pb is None:
            return None
        d_a, atoms_a, rows_a = pa
        d_b, atoms_b, rows_b = pb
        el = self.eq(rows_a[0][r], rows_b[0][r])
        er = self.eq(rows_a[1][r], rows_b[1][r])
        ed = self.eq(d_a, d_b)
        name = self.fresh()
        hyps_a = ' '.join(self.hyp('A', x) for x in atoms_a)
        hyps_b = ' '.join(self.hyp('B', x) for x in atoms_b)
        self.lines.append(f'  have {name} : {self.term("A", wa)} = {self.term("B", wb)} :=')
        self.lines.append(f'    Step0Agree.stack_slice_agree {r} {hyps_a} {hyps_b}')
        self.lines.append(f'      {self.one(el)} {self.one(er)} {self.one(ed)} '
                          f'{self.hyp("A", aa)} {self.hyp("B", ab)}')
        self.close(name, wa, wb)
        return name

    def atom_eq(self, wa, wb, aa, ab):
        req(aa.kind == ab.kind, f'atom kind {aa.kind} vs {ab.kind}')
        special = self.stack_slice_eq(wa, wb, aa, ab)
        if special is not None:
            return special
        req(shape(' '.join(aa.args)) == shape(' '.join(ab.args)), f'atom shape {aa.text} vs {ab.text}')
        nout = 2 if aa.kind == 'sha256BitColumns' else 1
        subs = self.collect(WIRE.findall(' '.join(aa.args[:-nout])),
                            WIRE.findall(' '.join(ab.args[:-nout])))
        name = self.fresh()
        h = self.fresh('hx')
        self.lines.append(f'  have {h} := {self.hyp("A", aa)}')
        if subs:
            self.lines.append(f'  rw [{", ".join(subs)}] at {h}')
        lhs, rhs = self.term('A', wa), self.term('B', wb)
        if aa.kind == 'sha256BitColumns':
            which = '.1' if strip_parens(aa.args[-2]) == wa else '.2'
            proof = f'(Step0Agree.sha256BitColumns_unique {h} {self.hyp("B", ab)}){which}'
        elif aa.kind == 'hashSample':
            proof = (f'(MxxExponentLut.hashSample_eq {h}).trans '
                     f'(MxxExponentLut.hashSample_eq {self.hyp("B", ab)}).symm')
        else:
            lemma = UNIQUE.get(aa.kind)
            req(lemma is not None, f'no uniqueness lemma for {aa.kind}')
            proof = f'{lemma} {h} {self.hyp("B", ab)}'
        self.lines.append(f'  have {name} : {lhs} = {rhs} := {proof}')
        self.close(name, wa, wb)
        return name

    # ------------------------------------------------------------------ loops
    def loop_eq(self, wa, wb, na, nb):
        aa, pa_ = na[1], na[2]
        ab, pb_ = nb[1], nb[2]
        req(pa_ == pb_, f'loop output paths differ {wa} {wb}')
        req(aa.loop.kind == ab.loop.kind, f'loop kinds differ {wa} {wb}')
        if aa.loop.kind == 'seq':
            return self.seq_eq(wa, wb, aa, ab, pa_)
        return self.member_eq(wa, wb, aa, ab, pa_)

    def arg_proof(self, ta, tb):
        """Equality proof of two corresponding child arguments (terms)."""
        ta, tb = strip_parens(ta), strip_parens(tb)
        ma = re.fullmatch(r'(w_\d+(?:_\d+)?(?:_gathered)?)(?: (\w+))?', ta)
        mb = re.fullmatch(r'(w_\d+(?:_\d+)?(?:_gathered)?)(?: (\w+))?', tb)
        if ma and mb:
            req(bool(ma.group(2)) == bool(mb.group(2)), f'arg application mismatch {ta} {tb}')
            if ma.group(1).endswith('_gathered'):
                return self.gathered[(ma.group(1), mb.group(1))]
            names = self.eq(ma.group(1), mb.group(1))
            req(len(names) == 1, f'arg {ta} has no single equality')
            return f'(congrFun {names[0]} {ma.group(2)})' if ma.group(2) else names[0]
        req(shape(ta) == shape(tb) and not WIRE.search(ta), f'arg terms {ta} {tb}')
        return 'rfl'

    def member_eq(self, wa, wb, aa, ab, path):
        ca, args_a, outs_a = call_args(aa)
        cb, args_b, outs_b = call_args(ab)
        child = self.gen.child_lemma(ca, cb, (path,))
        var = aa.loop.var
        name = self.fresh()
        lhs, rhs = self.term('A', wa), self.term('B', wb)
        if aa.loop.kind == 'par':
            proofs = [self.arg_proof(navigate(args_a, p), navigate(args_b, p))
                      for p in child.input_order()]
            self.lines.append(f'  have {name} : {lhs} = {rhs} :=')
            self.lines.append(f'    funext fun {var} ↦ {child.name} ({self.hyp("A", aa)} {var}) '
                              f'({self.hyp("B", ab)} {var}) {" ".join(proofs)}')
            self.close(name, wa, wb)
            return name
        # gather: destructure the gathered members first
        binders_a, conj_a = gather_conj(aa)
        binders_b, conj_b = gather_conj(ab)
        req(len(conj_a) == len(conj_b) and len(binders_a) == len(binders_b), 'gather shape')
        needed = {strip_parens(navigate(args_a, p)) for p in child.input_order()}
        lines = [f'  have {name} : {lhs} = {rhs} := by', f'    funext {var}']
        lines += self.destruct_lines([f'{g}A' for g in binders_a] + [f'gA{j}' for j in range(len(conj_a))],
                                     f'({self.hyp("A", aa)} {var})', 'gsA', '    ')
        lines += self.destruct_lines([f'{g}B' for g in binders_b] + [f'gB{j}' for j in range(len(conj_b))],
                                     f'({self.hyp("B", ab)} {var})', 'gsB', '    ')
        self.gathered = {}
        for j, c in enumerate(conj_a):
            h, args = app_split(c)
            if not h.endswith('familyGetDynamic'):
                continue
            g = strip_parens(args[2])
            if g not in needed:
                continue
            hb, argsb = app_split(conj_b[j])
            gb = strip_parens(argsb[2])
            names = self.collect(WIRE.findall(' '.join(args[:2])), WIRE.findall(' '.join(argsb[:2])))
            if names:
                lines.append(f'    rw [{", ".join(names)}] at gA{j}')
            lines.append(f'    have {g}E : {g}A = {gb}B := familyGetDynamic_unique gA{j} gB{j}')
            self.gathered[(g, gb)] = f'{g}E'
        proofs = [self.arg_proof(navigate(args_a, p), navigate(args_b, p)) for p in child.input_order()]
        lines.append(f'    exact {child.name} gA{len(conj_a) - 1} gB{len(conj_b) - 1} {" ".join(proofs)}')
        self.lines.extend(lines)
        self.close(name, wa, wb)
        return name

    @staticmethod
    def destruct_lines(names, src, tmp, indent):
        if len(names) == 1:
            return [f'{indent}have {names[0]} := {src}']
        out = []
        rest = src
        for k, nm in enumerate(names[:-1]):
            t = f'{tmp}{k}'
            out.append(f'{indent}obtain ⟨{nm}, {t}⟩ := {rest}')
            rest = t
        out.append(f'{indent}have {names[-1]} := {rest}')
        return out

    def seq_eq(self, wa, wb, aa, ab, path):
        La, Lb = aa.loop, ab.loop
        init_a, init_b = parse_tuple(La.init), parse_tuple(Lb.init)
        final_a = parse_tuple(La.final)
        req(str(La.count) == str(Lb.count), f'seq counts {La.count} {Lb.count}')
        name = self.fresh()
        lhs, rhs = self.term('A', wa), self.term('B', wb)
        if str(La.count) == '0':
            sub = self.eq(strip_parens(navigate(init_a, path)), strip_parens(navigate(init_b, path)))
            req(len(sub) == 1, 'seq init equality')
            self.lines.append(
                f'  have {name} : {lhs} = {rhs} := (congrArg (fun s ↦ s{path}) '
                f'(Step0Agree.iterRuns_zero_eq {self.hyp("A", aa)} rfl)).trans ({sub[0]}.trans '
                f'(congrArg (fun s ↦ s{path}) (Step0Agree.iterRuns_zero_eq {self.hyp("B", ab)} rfl)).symm)')
            self.close(name, wa, wb)
            return name
        ca, args_a, _ = call_args(aa)
        cb, args_b, _ = call_args(ab)
        state = {p for p, _ in leaves(final_a)}
        S = {path}
        while True:
            child = self.gen.child_lemma(ca, cb, tuple(sorted(S)))
            more = set()
            for p in child.input_order():
                t = strip_parens(navigate(args_a, p))
                if t.startswith('current'):
                    more.add(t[len('current'):])
            if more <= S:
                break
            S |= more
        order = sorted(S)
        req(set(order) <= state, 'seq state paths')

        def keys(var):
            items = [f'{var}{p}' for p in order]
            return items[0] if len(items) == 1 else '(' + ', '.join(items) + ')'

        def comp(var, p):
            k = order.index(p)
            if len(order) == 1:
                return var
            return f'{var}' + '.2' * k + ('.1' if k < len(order) - 1 else '')
        init_eqs = []
        for p in order:
            sub = self.eq(strip_parens(navigate(init_a, p)), strip_parens(navigate(init_b, p)))
            req(len(sub) == 1, 'seq init equality')
            init_eqs.append(sub[0])
        proofs = []
        for p in child.input_order():
            ta = strip_parens(navigate(args_a, p))
            if ta.startswith('current'):
                sp = ta[len('current'):]
                k = order.index(sp)
                if len(order) == 1:
                    proofs.append('hprev')
                else:
                    sel = '.2' * k + ('.1' if k < len(order) - 1 else '')
                    proofs.append(f'(congrArg (fun t ↦ t{sel}) hprev)')
            else:
                proofs.append(self.arg_proof(ta, navigate(args_b, p)))
        lines = [f'  have {name} : {lhs} = {rhs} := by',
                 f'    have key := iter_projected_agree (fun s ↦ {keys("s")}) (fun s ↦ {keys("s")})',
                 f'      ?_ ?_ {self.hyp("A", aa)} {self.hyp("B", ab)}']
        if len(order) == 1:
            lines.append('    · exact key')
        else:
            k = order.index(path)
            sel = '.2' * k + ('.1' if k < len(order) - 1 else '')
            lines.append(f'    · exact congrArg (fun t ↦ t{sel}) key')
        if len(order) == 1:
            lines.append(f'    · exact {init_eqs[0]}')
        else:
            lines.append(f'    · show ({", ".join(self.term("A", strip_parens(navigate(init_a, p))) for p in order)}) = '
                         f'({", ".join(self.term("B", strip_parens(navigate(init_b, p))) for p in order)})')
            lines.append(f'      rw [{", ".join(dict.fromkeys(init_eqs))}]')
        lines.append('    · intro digit lc ln rc rn hprev hl hr')
        lines.append(f'      have r := {child.name} hl hr {" ".join(proofs)}')
        if len(order) == 1:
            lines.append('      exact r')
        else:
            parts = []
            for j in range(len(order)):
                parts.append('r' + ('' if len(order) == 1 else
                             ('.2' * j + ('.1' if j < len(order) - 1 else ''))))
            lines.append(f'      exact {nested_pair(parts)}')
        self.lines.extend(lines)
        self.close(name, wa, wb)
        return name

    def input_order(self):
        return list(self.inputs)


def and_proj(index, count):
    if count == 1:
        return ''
    if index == count - 1:
        return '.2' * index
    return '.2' * index + '.1'


def nested_pair(items):
    if len(items) == 1:
        return items[0]
    return f'Prod.ext {items[0]} ({nested_pair(items[1:])})'


class Generator:
    def __init__(self, package, modA, modB):
        self.pkg = Path(package)
        self.modA, self.modB = modA, modB
        worst = str(self.pkg / 'worst')
        self.scA = xmodel.load(worst, modA)
        self.scB = xmodel.load(worst, modB)
        self.srcA = self.read(modA)
        self.srcB = self.read(modB)
        self.infoA, self.infoB = {}, {}
        self.lemmas = {}
        self.order = []
        self.counter = 0

    def read(self, mod):
        text = ''
        for base in (mod, mod + '_Facts'):
            for _, t in module_sources(self.pkg / 'worst', base):
                text += xmodel.expand_matrix_notations(t) + '\n'
        return text

    def info(self, side, name):
        cache = self.infoA if side == 'A' else self.infoB
        if name not in cache:
            scopes = self.scA if side == 'A' else self.scB
            mod = self.modA if side == 'A' else self.modB
            req(name in scopes, f'unknown scope {name}')
            src = self.srcA if side == 'A' else self.srcB
            m = re.search(r'^(?:noncomputable )?def ' + re.escape(name) + r'\.outs [^\n]*:=\n  ([^\n]*)',
                          src, re.M)
            cache[name] = ScopeInfo(mod, scopes[name], m.group(1) if m else None)
        return cache[name]

    def child_lemma(self, ca, cb, outs):
        key = (ca, cb, tuple(outs))
        if key in self.lemmas:
            return self.lemmas[key]
        A, B = self.info('A', ca), self.info('B', cb)
        req(A.record == B.record, f'scope kinds differ {ca} {cb}')
        da = re.sub(r'^def \S+', 'def S', A.s.def_src or '')
        db = re.sub(r'^def \S+', 'def S', B.s.def_src or '')
        req(shape(da) == shape(db), f'scope defs differ {ca} {cb}')
        self.counter += 1
        L = Lemma(self, A, B, outs, f'Step0Agree.agree_{self.counter}')
        self.lemmas[key] = L
        outsA = dict(A.outputs())
        outsB = dict(B.outputs())
        L.out_eqs = [L.eq(outsA[p], outsB[p]) for p in outs]
        self.order.append(L)
        return L

    def render_child(self, L):
        A, B = L.A, L.B
        xs = [f'x{j}' for j in range(len(A.s.binders))]
        binders = ('{' + ' '.join(xs) + ' : Nat} ') if xs else ''
        args = ' '.join(xs)
        short = L.name.split('.')[-1]
        head = (f'theorem {short} {{backend : BackendContext}} {{model : HashModel}}\n'
                f'    {{pA : {A.module}.Params}} {{pB : {B.module}.Params}} {binders}{{io oo ik ok : _}}\n'
                f'    (ho : {A.module}.{A.name} backend model pA {args} io oo)\n'
                f'    (hk : {B.module}.{B.name} backend model pB {args} ik ok)')
        for path, h in L.inputs.items():
            head += f'\n    ({h} : io{path} = ik{path})'
        concl = ' ∧ '.join(f'oo{p} = ok{p}' for p in L.outs)
        head += f' :\n    {concl} := by\n'
        body = []
        if A.record:
            body += ['  obtain ⟨wa, ba⟩ := ho', '  obtain ⟨wb, bb⟩ := hk',
                     f'  let co : {A.module}.{A.name}.Ctx backend model pA {args} := ⟨io, wa⟩',
                     f'  let ck : {B.module}.{B.name}.Ctx backend model pB {args} := ⟨ik, wb⟩',
                     f'  have Fo := {A.module}.{A.name}.facts co ba',
                     f'  have Fk := {B.module}.{B.name}.facts ck bb']
            outA = next(a for a in A.s.atoms if a.field == 'hout')
            outB = next(a for a in B.s.atoms if a.field == 'hout')
            body += L.lines
            body.append(f'  rw [Fo.c{outA.frag}.hout, Fk.c{outB.frag}.hout]')
            names = [e[0] for e in L.out_eqs]
        else:
            fr = A.frags
            body.append(f'  dsimp only [{A.module}.{A.name}] at ho')
            body.append(f'  dsimp only [{B.module}.{B.name}] at hk')
            body += Lemma.destruct_lines(A.exists + [f'cA{k}' for k in fr], 'ho', 'tA', '  ')
            body += Lemma.destruct_lines([x + "'" for x in B.exists] + [f'cB{k}' for k in fr], 'hk', 'tB', '  ')
            body += L.lines
            body.append(f'  rw [{L.hyp("A", A.out_atom)}, {L.hyp("B", B.out_atom)}]')
            names = [(f'(by rw [{", ".join(e)}])' if e else 'rfl') for e in L.out_eqs]
        if len(names) == 1:
            body.append(f'  exact {names[0]}')
        else:
            body.append('  exact ⟨' + ', '.join(names) + '⟩')
        return head + '\n'.join(body) + '\n'


def claim_tree(package, stage):
    text = (Path(package) / 'worst' / 'W_Claim.lean').read_text()
    for line in text.split('\n'):
        l = line.strip()
        if l.startswith(stage + '.generatedRoot '):
            head, args = app_split(l.rstrip(' ∧'))
            return parse_tuple(args[-2])
    raise AgreeError(f'no claim call for {stage}')


class RootLemma(Lemma):
    """The root agreement theorem: inputs agree when both claims read one source."""

    def __init__(self, gen):
        super().__init__(gen, gen.info('A', 'generatedRoot'), gen.info('B', 'generatedRoot'),
                         (), 'root', mode='root')
        self.claimA = claim_tree(gen.pkg, gen.modA)
        self.claimB = claim_tree(gen.pkg, gen.modB)
        self.root_inputs = {}

    def input_eq(self, pa, pb, wa, wb):
        key = (pa, pb)
        if key in self.root_inputs:
            return self.root_inputs[key]
        ca = strip_parens(xmodel.norm_wires(str(navigate(self.claimA, pa))))
        cb = strip_parens(xmodel.norm_wires(str(navigate(self.claimB, pb))))
        req(isinstance(navigate(self.claimA, pa), str) and ca == cb,
            f'root inputs {wa}{pa}={ca[:80]} and {wb}{pb}={cb[:80]} differ')
        name = f'r{len(self.root_inputs)}'
        self.lines.append(f'  have {name} : {self.term("A", wa)} = {self.term("B", wb)} := by')
        self.lines.append(f'    change co.inputs{pa} = ck.inputs{pb}')
        self.lines.append('    rw [hino, hink]; rfl')
        self.close(name, wa, wb)
        self.root_inputs[key] = name
        return name


NAME_REF = re.compile(r'\b([er]\d+)\b')


def root_chunks(R, binders, call_args, budget):
    """Split the root equalities into theorems of about `budget` lines.

    `binders` is the shared parameter text of every chunk theorem and
    `call_args` the corresponding argument text.  Returns the chunk theorem
    texts and the tactic lines that rebuild the exported names in the final
    theorem.  Each chunk takes exactly the earlier equalities it references.
    """
    req(R.mark == len(R.lines), 'root lines outside blocks')
    statement = {name: stmt for name, stmt, _ in R.blocks}
    chunks, current, size = [], [], 0
    for block in R.blocks:
        current.append(block)
        size += len(block[2])
        if size >= budget:
            chunks.append(current)
            current, size = [], 0
    if current:
        chunks.append(current)
    defined_in = {}
    for k, chunk in enumerate(chunks):
        for name, _, _ in chunk:
            defined_in[name] = k
    uses = []
    for chunk in chunks:
        refs = set()
        for name, _, lines in chunk:
            for line in lines:
                refs.update(m for m in NAME_REF.findall(line) if m in statement)
        uses.append(refs)
    return chunks, defined_in, uses, statement


def render_root(R, binders, call_args, budget, final_refs, prefix):
    chunks, defined_in, uses, statement = root_chunks(R, binders, call_args, budget)
    later = [set() for _ in chunks]
    for k in range(len(chunks)):
        for ref in uses[k]:
            if defined_in[ref] < k:
                later[defined_in[ref]].add(ref)
    for ref in final_refs:
        later[defined_in[ref]].add(ref)
    theorems, rebuild = [], []
    for k, chunk in enumerate(chunks):
        own = {name for name, _, _ in chunk}
        external = sorted((r for r in uses[k] if r not in own), key=lambda x: (x[0], int(x[1:])))
        exports = [name for name, _, _ in chunk if name in later[k]]
        hyps = ''.join(f'\n    ({r} : {statement[r]})' for r in external)
        concl = ' ∧ '.join(statement[name] for name in exports) or 'True'
        text = f'theorem {prefix}{k} {binders}{hyps} :\n    {concl} := by\n'
        for _, _, lines in chunk:
            text += '\n'.join(lines) + '\n'
        if not exports:
            text += '  trivial\n'
        elif len(exports) == 1:
            text += f'  exact {exports[0]}\n'
        else:
            text += '  exact ⟨' + ', '.join(exports) + '⟩\n'
        theorems.append(text)
        call = f'{prefix}{k} {call_args} {" ".join(external)}'.rstrip()
        if not exports:
            continue
        if len(exports) == 1:
            rebuild.append(f'  have {exports[0]} := {call}')
        else:
            rebuild.append(f'  obtain ⟨{", ".join(exports)}⟩ := {call}')
    return theorems, rebuild
