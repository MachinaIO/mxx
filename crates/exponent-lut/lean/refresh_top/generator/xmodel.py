#!/usr/bin/env python3
"""Structural model of a refresh_top export (`generated/`): scopes, loops, atoms, wires, facts.

For every module `M` (`Stage_*`, `Requirement_0`) the model reads `M.lean` (scope definitions,
`constraints_k` fragments, `Witness` structures) and `M_Facts.lean` (per record scope: `Ctx`,
wire definitions `S.w_<n>_<p>`, fragment records `S.Facts_k`, `_at` lemmas), and `Claim.lean`
(`Runs`: the stage calls and their argument tuples).

Scope model (`Scope`):
  name, module, binders (index variables `i_k` in order, `_` for unused), kind
  ('record' = `∃ witness, body`; 'chain' = `let`/`∃` chain ending in constraints calls),
  inputs (list of component types of the input tuple; [] for `Unit`, the type itself when not a
  tuple), output (list of component types), wires (ordered: name -> Wire(type, src, term)),
  atoms (ordered list of `Atom`), loops (the loop atoms), children (child scope names).
Atom: frag (fragment index), field (record field name or `a<k>` for chain scopes), head (constant
  of the predicate, e.g. `MxxRuntime.sliceMatrix`; `cmp`/`eq`/`out` for comparisons, equalities and
  the output equation), args (top-level argument terms, wire references normalized to `w_<n>_<p>`),
  wires (wires mentioned), text (normalized text), loop (Loop or None).
Loop: kind ('par'|'gather'|'seq'), count (int or str), child, args (child input tuple terms),
  outs (child output term(s)), gathered (gather: [(fam, idx, var)]), state (seq: init, final).

Usage: xmodel.py <generated dir> <Module> [scope]  -- prints a summary (scope tree + atoms).
"""
import collections
import json
import re
import sys

PAIRS = {'(': ')', '[': ']', '⟨': '⟩', '{': '}'}
CLOSE = set(PAIRS.values())


# --------------------------------------------------------------------------- term utilities

def split_top(s, sep):
    """Split `s` at top-level occurrences of the string `sep`."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch in PAIRS:
            depth += 1
        elif ch in CLOSE:
            depth -= 1
        if depth == 0 and s.startswith(sep, i):
            out.append(''.join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    out.append(''.join(cur))
    return out


def strip_parens(s):
    s = s.strip()
    while s.startswith('(') and matching(s, 0) == len(s) - 1:
        s = s[1:-1].strip()
    return s


def matching(s, i):
    depth = 0
    for j in range(i, len(s)):
        if s[j] in PAIRS:
            depth += 1
        elif s[j] in CLOSE:
            depth -= 1
            if depth == 0:
                return j
    return -1


def app_split(s):
    """Head and top-level arguments of an application `f a (b c) [d]`."""
    s = strip_parens(s)
    parts = [p for p in split_top(s, ' ') if p != '']
    return parts[0] if parts else '', parts[1:]


def tuple_items(s):
    """Items of a right-nested tuple `(a, (b, (c, ())))` (also flat `(a, b, c)`); `()` -> []."""
    s = s.strip()
    if s in ('()', ''):
        return []
    if not (s.startswith('(') and matching(s, 0) == len(s) - 1):
        if len(split_top(s, ',')) > 1:
            return tuple_items('(' + s + ')')
        return [s]
    inner = s[1:-1]
    parts = split_top(inner, ',')
    if len(parts) == 1:
        inner = inner.strip()
        if inner.startswith('(') and matching(inner, 0) == len(inner) - 1 and \
                len(split_top(inner[1:-1], ',')) > 1:
            return tuple_items(inner)
        return [strip_parens(inner)] if inner != '' else []
    items = [p.strip() for p in parts[:-1]]
    last = parts[-1].strip()
    if len(parts) == 2:
        return [strip_parens(items[0])] + tuple_items(last)
    return [strip_parens(x) for x in items] + tuple_items(last)


def type_items(t):
    """Components of a product type `A × B × … × Unit` (trailing `Unit` dropped)."""
    t = strip_parens(t)
    if t == 'Unit':
        return []
    parts = [strip_parens(p) for p in split_top(t, ' × ')]
    if parts and parts[-1] == 'Unit':
        parts = parts[:-1]
    return parts


WIREREF = re.compile(r'\(([A-Za-z_0-9]+)\.(w_\d+(?:_\d+)?(?:_concat_\d+)?) ctx\)')


def norm_wires(s):
    """`(S.w_3_0 ctx)` -> `w_3_0`."""
    return WIREREF.sub(r'\2', s)


WIRE = re.compile(r'\bw_\d+(?:_\d+)?(?:_concat_\d+)?(?:_gathered)?\b')


# --------------------------------------------------------------------------- model types

class Wire:
    __slots__ = ('name', 'type', 'src', 'term')

    def __init__(self, name, type_, src, term=None):
        self.name, self.type, self.src, self.term = name, type_, src, term

    def to_json(self):
        return dict(type=self.type, src=self.src, term=self.term)


class Loop:
    __slots__ = ('kind', 'count', 'child', 'args', 'outs', 'gathered', 'init', 'final', 'var')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def to_json(self):
        return {k: getattr(self, k) for k in self.__slots__}


class Atom:
    __slots__ = ('frag', 'field', 'head', 'args', 'wires', 'text', 'loop')

    def __init__(self, frag, field, text):
        self.frag, self.field = frag, field
        self.text = norm_wires(text.strip())
        self.loop = parse_loop(self.text)
        self.head, self.args = classify(self.text, self.loop)
        self.wires = list(dict.fromkeys(WIRE.findall(self.text)))

    @property
    def kind(self):
        return self.head.split('.')[-1]

    def to_json(self):
        return dict(frag=self.frag, field=self.field, head=self.head, args=self.args,
                    wires=self.wires, loop=self.loop.to_json() if self.loop else None)


class Scope:
    def __init__(self, module, name):
        self.module, self.name = module, name
        self.binders, self.kind = [], None
        self.inputs, self.output, self.input_type, self.output_type = [], [], None, None
        self.wires = collections.OrderedDict()
        self.atoms, self.at = [], []
        self.uses_backend = self.uses_hash = False
        self.params = []  # leading parameter names of the def (backend, model, params)
        self.def_src = self.body_src = None   # raw `def S` and `S.body` texts
        self.frag_src = {}                     # k -> (binders [(name, type)], body text)
        self.witness = []                      # [(field, type)] of `S.Witness`
        self.sig = []                          # [(name, type)] parameters of `def S`

    @property
    def loops(self):
        return [a for a in self.atoms if a.loop]

    @property
    def children(self):
        return list(dict.fromkeys(a.loop.child for a in self.loops if a.loop.child))

    def atom_kinds(self):
        return collections.Counter(a.kind for a in self.atoms
                                   if a.head not in ('cmp', 'eq', 'out', 'true') and not a.loop)

    def to_json(self):
        return dict(module=self.module, name=self.name, binders=self.binders, kind=self.kind,
                    inputs=self.inputs, output=self.output, params=self.params,
                    wires={k: w.to_json() for k, w in self.wires.items()},
                    atoms=[a.to_json() for a in self.atoms], at=self.at,
                    children=self.children)


# --------------------------------------------------------------------------- atoms and loops

CHILD = re.compile(r'\b((?:parallel|sequential)_[A-Za-z_0-9]+) (?:backend |params |\()')
HEAD = re.compile(r'^((?:Mxx\w*\.)+\w+|[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_0-9]+)*)')


def child_call(text):
    """The (last) scope call `child b m p i… (args) (outs)` inside `text`."""
    m = None
    for m in re.finditer(r'\b((?:parallel|sequential)_[A-Za-z_0-9]+) ', text):
        pass
    if not m:
        return None
    call = text[m.start():]
    head, args = app_split(call)
    return head, args


def parse_loop(t):
    s = strip_parens(t)
    m = re.match(r'^∀ (\w+) : Fin (\d+), ', s)
    if m:
        var, count, body = m.group(1), int(m.group(2)), s[m.end():]
        gathered = []
        if body.startswith('∃ '):
            # gather: ∃ x : T, ∃ y : T, conj ∧ … ∧ child …
            binders = []
            while body.startswith('∃ '):
                mm = re.match(r'^∃ (\w+) : ', body)
                rest = body[mm.end():]
                ty = split_top(rest, ', ')[0]
                binders.append(mm.group(1))
                body = rest[len(ty) + 2:]
            conj = [c.strip() for c in split_top(body, ' ∧ ')]
            for c in conj:
                h, a = app_split(c)
                if h.endswith('familyGetDynamic'):
                    gathered.append((norm_wires(a[0]), strip_parens(a[1]), a[2]))
            call = conj[-1]
            kind = 'gather'
        else:
            call = body
            kind = 'par'
        head, args = app_split(call)
        return Loop(kind=kind, count=count, child=head, args=tuple_items(args[-2]),
                    outs=tuple_items(args[-1]), gathered=gathered, var=var)
    if s.startswith('MxxIR.IterRuns'):
        head, args = app_split(s)
        # args: (fun (i : Nat) (current next : T) => child … (tuple) next) (Int.toNat (n)) init final
        fun = strip_parens(args[0])
        body = fun.split('=> ', 1)[1]
        ch, cargs = app_split(body)
        cnt = strip_parens(args[1])
        mm = re.match(r'^Int\.toNat \((.*)\)$', cnt)
        cnt = strip_parens(mm.group(1)) if mm else cnt
        try:
            cnt = int(cnt)
        except ValueError:
            pass
        return Loop(kind='seq', count=cnt, child=ch, args=tuple_items(cargs[-2]), outs=[cargs[-1]],
                    init=strip_parens(args[2]), final=strip_parens(args[3]), gathered=[])
    return None


def classify(t, loop):
    if loop:
        return 'loop:' + loop.kind, []
    s = strip_parens(t)
    if s.startswith('outputs = '):
        return 'out', [s[len('outputs = '):]]
    m = HEAD.match(s)
    if m and (m.group(1).startswith('Mxx') or m.group(1).startswith('Mxx')):
        h, a = app_split(s)
        return h, a
    if s == 'True':
        return 'true', []
    if ' = ' in s and '≤' not in s and ' < ' not in s and '≠' not in s:
        return 'eq', split_top(s, ' = ')
    return 'cmp', [s]


# --------------------------------------------------------------------------- module parsing

DECL = re.compile(r'^(?:noncomputable )?(def|abbrev|structure|theorem|set_option)\s+([^\s(]+)', re.M)


def split_decls(text):
    ms = list(DECL.finditer(text))
    out = []
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
        out.append((m.group(1), m.group(2), text[m.start():end]))
    return out


def binders_of(sig):
    """Parameters of a scope definition signature, in order: (name, type)."""
    out = []
    i = sig.index(' ', sig.index('def ') + 4) if 'def ' in sig else 0
    s = sig[i:]
    for m in re.finditer(r'\(', s):
        pass
    depth = 0
    j = 0
    while j < len(s):
        if s[j] == '(' and depth == 0:
            k = matching(s, j)
            inner = s[j + 1:k]
            if ' : ' in inner:
                n, t = inner.split(' : ', 1)
                out.append((n.strip(), t.strip()))
            j = k + 1
            continue
        if s[j] == ':' and depth == 0 and s[j:j + 3] == ': P':
            break
        j += 1
    return out


def split_conj(body):
    return [c.strip() for c in split_top(body.strip(), ' ∧\n')]


def parse_module(gdir, module):
    text = open(f'{gdir}/{module}.lean').read()
    scopes = collections.OrderedDict()
    frags = collections.defaultdict(dict)   # scope -> k -> (params, body)
    defs = {}
    witness = {}
    for kind, name, src in split_decls(text):
        if kind == 'set_option':
            continue
        base = name.split('.')[0]
        if base.startswith(('sel_', 'mod_', 'modulus_', 'Params')) or base in ('Params',):
            continue
        if kind == 'structure' and name.endswith('.Witness'):
            witness[base] = re.findall(r'^\s+(w_\d+(?:_\d+)?(?:_concat_\d+)?) : (.*)$', src, re.M)
            continue
        m = re.match(r'.*\.constraints_(\d+)$', name)
        if m:
            sig, body = src.split(' : Prop :=\n', 1)
            frags[base][int(m.group(1))] = (binders_of(sig), body)
            continue
        if name.endswith('.body'):
            defs.setdefault(base, {})['body'] = src
            continue
        if kind == 'def' and '.' not in name:
            defs.setdefault(base, {})['def'] = src
    for base, d in defs.items():
        if 'def' not in d:
            continue
        sc = Scope(module, base)
        sig, body = d['def'].split(' : Prop :=\n', 1)
        params = binders_of(sig)
        sc.params = [n for n, t in params if t in ('MxxRuntime.BackendContext', 'MxxRuntime.HashModel', 'Params')
                     or n == '_' and t in ('MxxRuntime.HashModel',)]
        sc.binders = [n for n, t in params if t == 'Nat']
        tin = dict(params).get('inputs')
        tout = dict(params).get('outputs')
        sc.input_type, sc.output_type = tin, tout
        sc.inputs = type_items(tin) if tin else []
        sc.output = type_items(tout) if tout else []
        sc.kind = 'record' if base in witness else 'chain'
        sc.def_src, sc.body_src = d['def'], d.get('body')
        sc.frag_src = dict(frags[base])
        sc.witness = witness.get(base, [])
        sc.sig = params
        src_body = d.get('body', d['def'])
        # wires from let / ∃ / witness
        for m in re.finditer(r'let (w_\d+(?:_\d+)?(?:_concat_\d+)?)(?: : (.*?))? := (.*)$', src_body, re.M):
            nm, ty, term = m.group(1), m.group(2), m.group(3).strip()
            if term.startswith('inputs'):
                sc.wires[nm] = Wire(nm, ty, 'input', term)
            elif term.startswith('witness.'):
                sc.wires[nm] = Wire(nm, dict(witness[base]).get(nm), 'witness', term)
            else:
                sc.wires[nm] = Wire(nm, ty, 'let', term)
        for m in re.finditer(r'∃ \((w_\d+(?:_\d+)?(?:_concat_\d+)?) : (.*)\),\s*$', src_body, re.M):
            sc.wires[m.group(1)] = Wire(m.group(1), m.group(2), 'exists')
        for k in sorted(frags[base]):
            _, fb = frags[base][k]
            for j, c in enumerate(split_conj(fb)):
                sc.atoms.append(Atom(k, f'c{k}_{j}', c))
        scopes[base] = sc
    return scopes


def parse_facts(gdir, module, scopes):
    """Attach record fields (names), wire definitions and `_at` lemmas from `M_Facts.lean`."""
    try:
        text = open(f'{gdir}/{module}_Facts.lean').read()
    except FileNotFoundError:
        return
    for m in re.finditer(r'^def ([A-Za-z_0-9]+)\.(w_\d+(?:_\d+)?(?:_concat_\d+)?) \{[^\n]*\) : ([^\n]*) :=\n  ([^\n]*)', text, re.M):
        sc = scopes.get(m.group(1))
        if sc is None:
            continue
        term = norm_wires(m.group(4).strip())
        w = sc.wires.get(m.group(2))
        if w is None:
            sc.wires[m.group(2)] = Wire(m.group(2), m.group(3), 'let', term)
        else:
            w.type = w.type or m.group(3)
    for m in re.finditer(r'^structure ([A-Za-z_0-9]+)\.Facts_(\d+) [^\n]*\n((?:  [^\n]*\n)*)', text, re.M):
        sc = scopes.get(m.group(1))
        if sc is None:
            continue
        k = int(m.group(2))
        fields = [line.strip().split(' : ', 1) for line in m.group(3).splitlines()]
        old = [a for a in sc.atoms if a.frag == k]
        new = [Atom(k, f, body) for f, body in fields]
        if len(old) != len(new):
            # the record splits fragments the same way; keep the facts version
            pass
        sc.atoms = [a for a in sc.atoms if a.frag != k] + new
        sc.atoms.sort(key=lambda a: a.frag)
        sc.kind = 'record'
    for m in re.finditer(r'^theorem ([A-Za-z_0-9]+)\.(h\w+)_at \{', text, re.M):
        sc = scopes.get(m.group(1))
        if sc is not None:
            sc.at.append(m.group(2))
    # record fragment counts (Facts has c0..cK)
    for m in re.finditer(r'^structure ([A-Za-z_0-9]+)\.Facts \{[^\n]*\n((?:  [^\n]*\n)*)', text, re.M):
        sc = scopes.get(m.group(1))
        if sc is not None:
            sc.record_fields = [l.strip().split(' : ')[0] for l in m.group(2).splitlines()]


def load(gdir, module):
    sc = parse_module(gdir, module)
    parse_facts(gdir, module, sc)
    return sc


def roots(scopes):
    called = {c for s in scopes.values() for c in s.children}
    return [n for n in scopes if n not in called]


# --------------------------------------------------------------------------- claim

def parse_claim(gdir):
    """`Runs`: list of (module, root, args terms, output term)."""
    text = open(f'{gdir}/Claim.lean').read()
    body = text.split('def Runs', 1)[1].split(':=\n', 1)[1].split('\n\n')[0]
    calls = []
    for c in split_top(body.strip(), ' ∧\n'):
        c = c.strip()
        m = re.match(r'^(\w+)\.generatedRoot ', c)
        if not m:
            calls.append(dict(module=None, text=c))
            continue
        head, args = app_split(c)
        calls.append(dict(module=m.group(1), args=tuple_items(args[-2]), output=args[-1],
                          params=args[:-2]))
    return calls


# --------------------------------------------------------------------------- printing

def show(scopes, name, depth=0, how='root'):
    s = scopes[name]
    kinds = ' '.join(f'{k}:{n}' for k, n in s.atom_kinds().most_common())
    print('  ' * depth + f'{how} {name} [{s.kind}] in={len(s.inputs)} out={len(s.output)} | {kinds}')
    for a in s.loops:
        if a.loop.child in scopes:
            show(scopes, a.loop.child, depth + 1, f'{a.loop.kind}[{a.loop.count}]')


if __name__ == '__main__':
    gdir, mod = sys.argv[1], sys.argv[2]
    if mod == 'Claim':
        for c in parse_claim(gdir):
            print(c.get('module'), len(c.get('args', []) or []))
        sys.exit()
    sc = load(gdir, mod)
    if len(sys.argv) > 3:
        print(json.dumps(sc[sys.argv[3]].to_json(), indent=1, ensure_ascii=False))
    else:
        for r in roots(sc):
            show(sc, r)
