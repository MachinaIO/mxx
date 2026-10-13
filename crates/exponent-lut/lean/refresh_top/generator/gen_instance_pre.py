#!/usr/bin/env python3
"""Generate the preprocessing key facts `Gen_Inst_Stage_preprocess_Keys.lean` (L3, WP F).

From the facts record `F` of the preprocessing instance body (symbolic instance `i_0`), derive the
semantic facts the encrypt / online stages consume, as named theorems:

* `keys_<field>` (`prf.key-cell0/1` par loops): every cell's keys `cellKeyG …`, its tops
  `topPairG …` and its right-hand side `σ top_j + bot_j = (pp · message c e · u_j) G + e_j`
  (bridges `PrfKeyCell0.rel` / `PrfKeyCell1.rel`);
* `rhs_<field>`: the right-hand sides of all cells with ONE error family
  `eB : Fin N → Fin 2 → ErrorMatrix …` — the form of the online cell gathers' `hrhs` / `heB`;
* `table_<field>` (`lut.key-table` par loops): tops and `σ(s) top_j + bot_j = (T u_j) G + e_j`
  (`LutKeyTable.rel*`);
* `member_<field>` (`lut.key-member.*`, `smudge.key-member` par loops): the member's keys as a sum
  of the branch keys `lutKeysG …` of its child key loop (`LutKey*.rel` + `<Child>.hstep`);
* `keyloop_<field>` (`lut.key-round.step` seq loops at the instance root, from zero keys): the
  final keys `= ∑ k < count, lutKeysG … k` (`key_loop_of`);
* `preimage_<field>` (`preimageRunsDispatched` with its `trapdoorSample`): `A P = target`, `P` small;
* `link_<a>_<b>`: two hash samples of the instance under the same tags and key are equal
  (`Gen.hashSample_unique`).

Statements are the bridge lemmas' conclusions instantiated at the instance: the bridge's bundle
parameters (the `X.Comp` arguments before `backend`) are matched POSITIONALLY with the adapter's
(`<scope>.sem` in `Gen_<Stage>_Scopes.lean`), index parameters (`i0`, `i`; aligned from the right,
so `i0` may be absent) with `i_0` and the loop index, and the bundle's input / output tuples with the
loop's arguments — no instance constant (widths, moduli, tags, counts) is written here. Instance
hypotheses of the bridges are discharged generically: closed numeric side conditions by `decide` /
the `Gen_InstCerts` certificates / `rfl`; `sec = s`, `one = 1` facts by `xsem.MaskFacts` (secret
inputs = instance inputs fed by `Stage_payload` / `Stage_mask`); child-loop step hypotheses
(`KeyStepHyp …`) by `<ChildComp>.hstep` with the child's adapter. Everything else (gadgets, rebased
gadgets, free scales) becomes a hypothesis of the generated theorem.
Roles come from roles.json (xroles); bridges are chosen by the adapter's `Comp` name
(`X.Comp_v` -> `X.rel_v`, `X.hstep_v`) — never by wire ids or scope numbers.

Usage: gen_instance_pre.py <generated dir> <out dir> <param dir (adapters, bridges)> roles.json
       [Stage_preprocess] [--only field,field] [--out file]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import xmodel  # noqa: E402
import xsem  # noqa: E402
import gen_instance  # noqa: E402

# --------------------------------------------------------------------------- configuration
# Bridge files (in the parameter directory) searched for `X.rel` / `X.hstep`.
BRIDGES = ['GenBridge_Prf.lean', 'GenBridge_Lut.lean', 'GenBridge_LutKey.lean']
CELL_ROLES = ('prf.key-cell0', 'prf.key-cell1')
TABLE_ROLES = ('lut.key-table',)
MEMBER_ROLES = ('lut.key-member.readout', 'lut.key-member.sampler', 'lut.key-member.carrier',
                'smudge.key-member')
KEYLOOP_ROLES = ('lut.key-round.step',)
# Variables of the bridges' `variable {n : Nat}` sections (not in the theorem binders).
SECTION_VARS = {'n': 'Nat'}

IDENT = re.compile(r"(?<![\w.'])([^\W\d][\w']*)")
NAME = re.compile(r"^[^\W\d][\w']*$")


# --------------------------------------------------------------------------- Lean text parsing

def groups(s):
    """Top-level binder groups and the conclusion of a theorem header `binders : concl`."""
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '({[':
            j = xmodel.matching(s, i)
            out.append((ch, s[i + 1:j]))
            i = j + 1
            continue
        if ch == ':':
            return out, s[i + 1:].strip()
        raise ValueError(f'unexpected {s[i:i+40]!r}')
    raise ValueError('no conclusion')


def parse_binder(kind, text):
    """`{a b : T}` -> (kind, [a, b], T)."""
    names, ty = text.split(' : ', 1) if ' : ' in text else (text, None)
    return kind, names.split(), ty.strip() if ty else None


def parse_theorem(src, name):
    """Header of `theorem name … := …` in `src`: binders [(kind, names, type)] and conclusion."""
    m = re.search(r'^theorem ' + re.escape(name) + r' ', src, re.M)
    if not m:
        return None
    rest = src[m.end():]
    depth, i = 0, 0
    while i < len(rest):
        ch = rest[i]
        if ch in xmodel.PAIRS:
            depth += 1
        elif ch in xmodel.CLOSE:
            depth -= 1
        elif depth == 0 and rest.startswith(' :=', i):
            break
        i += 1
    head = ' '.join(rest[:i].split())
    gs, concl = groups(head)
    return dict(binders=[parse_binder(k, t) for k, t in gs], concl=concl)


def comp_call(ty):
    """`X.Comp toks… backend hashModel idx… [relation params…] inputs outputs`
    -> (comp, consts, mid (indices and relation parameters), None, inputs, outputs)."""
    head, args = xmodel.app_split(' '.join(ty.split()))
    k = args.index('backend')
    consts = args[:k]
    tail = args[k + 2:]
    ins, outs = tail[-2], tail[-1]
    return head, consts, tail[:-2], None, ins, outs


NAMED_ARG = re.compile(r"(?<![\w.'])([^\W\d][\w']*)(?= :=)")


def subst(text, mp):
    """Replace identifier tokens of `text` by their values in `mp` (not the names of named
    arguments `(x := …)`)."""
    def rep(m):
        if text.startswith(' :=', m.end()):
            return m.group(0)
        return mp.get(m.group(1), m.group(0))
    return IDENT.sub(rep, text)


def idents(text):
    return set(IDENT.findall(text))


def binder_types(binders):
    t = dict(SECTION_VARS)
    for _, names, ty in binders:
        if ty is not None:
            for n in names:
                t[n] = ty
    return t


def const_map(bridge_consts, adapter_consts, types):
    """Bridge parameter name -> instance value (type-ascribed), matched positionally."""
    if len(bridge_consts) != len(adapter_consts):
        raise ValueError(f'Comp arity {len(bridge_consts)} vs {len(adapter_consts)}')
    mp = {}
    for b, a in zip(bridge_consts, adapter_consts):
        a = xmodel.strip_parens(a)
        b = b.strip()
        mm = re.match(r'^\[(\w+)\]$', b)
        if mm:  # one-element list parameter `[p]`
            b = mm.group(1)
            if not (a.startswith('[') and a.endswith(']')):
                raise ValueError(f'list parameter {b} = {a}')
            a = xmodel.strip_parens(a[1:-1])
        if not NAME.match(b):
            continue  # a literal of the bridge
        t = types.get(b)
        v = f'(({a}) : {t})' if t and t != 'Nat' else f'({a})'
        if b in mp and mp[b] != v:
            raise ValueError(f'parameter {b}: {mp[b]} vs {v}')
        mp[b] = v
    return mp


# --------------------------------------------------------------------------- bridges and adapters

class Bridges:
    def __init__(self, pdir):
        self.src = '\n'.join(open(os.path.join(pdir, b)).read() for b in BRIDGES)
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            self.cache[name] = parse_theorem(self.src, name)
        if self.cache[name] is None:
            raise KeyError(f'bridge {name} not found')
        return self.cache[name]


def adapters(path):
    """scope -> (comp name, const tokens) of the bundle adapters `<scope>.sem : X.Comp …`;
    `adapters.mid[scope]` = (index / relation tokens after `backend hashModel`, Nat binder names)."""
    text = open(path).read()
    out = {}
    adapters.mid = {}
    for m in re.finditer(r'^theorem ([\w.]+)\.sem (.*)\n.*\n    ((\w+)\.(Comp\w*) .*) :=$', text, re.M):
        comp, consts, mid, _, _, _ = comp_call(m.group(3))
        out[m.group(1)] = (comp, consts)
        adapters.mid[m.group(1)] = (mid, re.findall(r'\{(\w+) : Nat\}', m.group(2)))
    return out


def inst_fun(C, f):
    """An adapter's relation argument `(fun x y ↦ S … i_0 i_1 x y)` at the instance: binders renamed
    apart, the adapter's index names replaced by `i_0` / `(i : Nat)`."""
    m = re.match(r'^\(fun (.*?) ↦ (.*)\)$', f, re.S)
    if not m:
        return f
    names = m.group(1).split()
    ren = {x: f'{x}_r' for x in names}
    body = subst(m.group(2), ren)
    return f'(fun {" ".join(ren[x] for x in names)} ↦ {body})'


def index_terms(C, scope):
    """The adapter's index tokens as instance terms (outer instance `i_0`, member `(i : Nat)`,
    deeper loop variables `None` = bound by the bridge) and its relation-parameter funs."""
    mid, nats = C.adm[scope]
    names = {}
    for j, n in enumerate(nats):
        names[n] = 'i_0' if j == 0 and len(nats) >= 2 else '((i : Nat))' if j == (1 if len(nats) >= 2 else 0) else None
    idx = [names.get(x.strip('()'), None) for x in mid if not x.startswith('(fun')]
    funs = [x for x in mid if x.startswith('(fun')]
    return idx, funs


def struct_adapters(path):
    """scope -> (conclusion, Nat binder names, hash-model binder name) of the structural adapters
    (`<scope>.sem : <Role>.Spec …`)."""
    text = open(path).read()
    out = {}
    for m in re.finditer(r'^theorem ([\w.]+)\.sem (.*)\n    \(h : .*\) :\n    (.*?) :=(?: by)?$', text, re.M):
        concl = m.group(3)
        if '.Comp' in concl.split(' ')[0]:
            continue
        nats = re.findall(r'\{(\w+) : Nat\}', m.group(2))
        hm = re.findall(r'\{(\w+) : MxxRuntime\.HashModel\}', m.group(2))
        out[m.group(1)] = (concl, nats, hm[0] if hm else 'hashModel')
    return out


def bridge_names(comp):
    """`X.Comp_2` -> (`X`, `_2`)."""
    base, v = comp.split('.Comp')
    return base, v


# --------------------------------------------------------------------------- instance context

class Ctx:
    def __init__(self, gdir, pdir, module, roles):
        self.module = module
        self.scopes = xmodel.load(gdir, module)
        root = self.scopes['generatedRoot'].to_json()
        rl = [a for a in root['atoms'] if a.get('loop') and a['loop']['kind'] == 'par'][0]['loop']
        self.inst, self.S = rl['child'], rl['count']
        text = open(f'{gdir}/{module}_Facts.lean').read()
        self.fields = gen_instance.facts_fields(text, self.inst)
        self.ij = self.scopes[self.inst].to_json()
        self.I = xsem.Inst(self.ij, self.fields, self.inst)
        self.roles = roles
        self.ad = adapters(f'{pdir}/Gen_{module}_Scopes.lean')
        self.adm = dict(adapters.mid)
        self.B = Bridges(pdir)
        self.sad = struct_adapters(f'{pdir}/Gen_{module}_Scopes.lean')
        self.certs = {}
        self.keyloops = {}
        self.secrets = {}
        for w, m in secret_inputs(gdir, module).items():
            s = 'sm' if m == 'Stage_mask' else 'sp'
            _, q, nn, _, _ = xsem.mat_type(self.ij['wires'][w]['type'])
            self.secrets[w] = (s, xsem.qterm(q), nn)

    def W(self, w, idx=None):
        return f'({self.inst}.{w} ctx{" " + idx if idx else ""})'

    def gadget_cert(self, q, nn, m, B, D):
        """Name (applied) of a certificate `∃ G : ExactMatrix q n 1 m, gadgetMatrixRuns Backend.backend B D G`
        (emitted once per gadget shape; per-instance moduli by `interval_cases` over the instances)."""
        key = (xmodel.strip_parens(q), nn, m, B, D)
        if key not in self.certs:
            self.certs[key] = f'{getattr(self, "cert_prefix", "gcert_")}{len(self.certs)}'
        inst = 'i_0' in key[0]
        return f'({self.certs[key]} i_0)' if inst else self.certs[key]

    def cert_text(self):
        out = []
        for (q, nn, m, B, D), name in self.certs.items():
            ty = f'∃ G : ExactMatrix {xsem.qterm(q)} {nn} 1 {m}, gadgetMatrixRuns Backend.backend {B} {D} G'
            pf = 'gadget_exists ⟨_, by simp [Backend.backend]; rfl, rfl, rfl, rfl⟩'
            if 'i_0' in q:
                out.append(f'theorem {name} (i_0 : Nat) [h : Fact (i_0 < S)] :\n    {ty} := by\n'
                           f'  have := h.out\n  interval_cases i_0 <;> exact {pf}\n')
            else:
                out.append(f'theorem {name} :\n    {ty} :=\n  {pf}\n')
        return '/-- Backend gadget certificates (layout base / digit count / width). -/\n' + '\n'.join(out) if out else ''

    def arg(self, a, idx='i'):
        """A loop argument `w_3_0` / `w_3_0 i` as a term."""
        parts = a.strip().split()
        w = xsem.wire_of(parts[0])
        if w is None:
            raise ValueError(f'argument {a}')
        return self.W(w, idx if len(parts) > 1 else None)


class Inst:
    """One bridge lemma at a bundle scope of the instance: parameter map `mp`."""

    def __init__(self, C, bridge_name, comp_consts, ins, outs, idx_vals, funs=()):
        th = C.B.get(bridge_name)
        self.name, self.th = bridge_name, th
        self.types = binder_types(th['binders'])
        h = [b for b in th['binders'] if b[0] == '(' and b[1] == ['h']][0]
        _, consts, mid, _, bins, bouts = comp_call(h[2])
        self.mp = const_map(consts, comp_consts, self.types)
        hm = xmodel.app_split(' '.join(h[2].split()))[1]
        k = hm.index('backend')
        self.mp['backend'], self.mp[hm[k + 1]] = 'backend', 'hashModel'
        bfun = [b for b in mid if '→' in self.types.get(b, '')]
        bidx = [b for b in mid if b not in bfun]
        for b, v in zip(reversed(bidx), reversed(idx_vals)):
            if NAME.match(b) and v is not None:
                self.mp[b] = v
        self.funs = dict(zip(bfun, funs))
        self.fun_terms = {b: inst_fun(C, f) for b, f in self.funs.items()}
        self.fun_idx = {}
        self.bind_tuple(bins, ins)
        self.bind_tuple(bouts, outs)

    def bind_tuple(self, pat, vals):
        if NAME.match(pat):
            self.mp[pat] = '(' + ', '.join(vals) + ', ())'
            return
        items = xmodel.tuple_items(pat)
        if len(items) != len(vals):
            raise ValueError(f'tuple {pat} vs {len(vals)} values')
        for p, v in zip(items, vals):
            if NAME.match(p):
                self.mp[p] = v

    def hyps(self):
        return [(n, ty) for k, ns, ty in self.th['binders'] if k == '(' and ns != ['h'] for n in ns]

    def implicits(self):
        return [(n, ty) for k, ns, ty in self.th['binders'] if k == '{' for n in ns]

    def free(self):
        return {n for n, _ in self.implicits()} - set(self.mp)



def secret_inputs(gdir, module):
    """instance input wires fed by the payload / mask stage outputs (the ternary secrets), through the
    Claim's stage wiring (`gen_stage_links.Stage.sources`; nested argument tuples)."""
    import gen_stage_links as SL
    calls = [c for c in xmodel.parse_claim(gdir) if c.get('module')]
    outs = {c['output'].strip(): c['module'] for c in calls}
    call = next(c for c in calls if c['module'] == module)
    st = SL.Stage(gdir, call)
    res = {}
    for w, (_, _, term, idx) in st.sources().items():
        if not idx and outs.get(term.strip()) in ('Stage_mask', 'Stage_payload'):
            res[w] = outs[term.strip()]
    return res


class Masks:
    """`xsem.MaskFacts` plus the secret inputs a theorem uses."""

    def __init__(self, C):
        self.C = C
        self.M = xsem.MaskFacts(C.I, {w: (f'h_{w}', f'h{s}small') for w, (s, _, _) in C.secrets.items()})
        self.used = set()

    def root(self, w):
        while w not in self.C.secrets:
            w = xsem.wire_of(self.C.I.prod[w]['args'][0])
        return w

    def fact(self, term):
        """(kind, proof) of `W w = …` for a wire term `(inst.w ctx)`, or (None, None)."""
        mm = re.match(r'^\(\w+\.(w_\d+_\d+) ctx\)$', term.strip())
        if not mm:
            return None, None
        w = mm.group(1)
        k = self.M.kind(w)
        if k is None:
            return None, None
        f = self.M.fact(w)
        if f is None:
            return None, None
        if k == 'sec':
            self.used.add(self.root(w))
        return k, f

    def binders(self):
        out = ''
        for w in sorted(self.used):
            s, q, nn = self.C.secrets[w]
            out += (f'\n    {{{s} : ErrorPoly {nn}}} (h{s}small : polyNorm {s} ≤ 1)'
                    f' (h_{w} : {self.C.inst}.{w} ctx = fun _ _ ↦ reducePoly {q} {nn} {s})')
        return out

    def args(self):
        return ' '.join(f'h{self.C.secrets[w][0]}small h_{w}' for w in sorted(self.used))


def discharge(C, Mk, free, t):
    """Proof text of the (substituted) hypothesis `t`, or None (the hypothesis is passed through)."""
    t = ' '.join(t.split())
    m = re.match(r'^(\(\w+\.w_\d+_\d+ ctx\)) = (.*)$', t)
    if m:
        k, f = Mk.fact(m.group(1))
        rhs = m.group(2).strip()
        if k == 'one':
            if rhs == '1':
                return f
            if re.match(r'^fun _ _ ↦ reducePoly .* 1$', rhs):
                return 'one_matrix_eq' if f == 'rfl' else f'({f}.trans one_matrix_eq)'
        if k == 'sec' and rhs.startswith('fun _ _ ↦ reducePoly'):
            return f
        return None
    if idents(t) & free:
        return None
    if re.match(r'^\(?\d+\)? ≤ [\d\s()*+]+$', t) or re.match(r'^\(?\d+\)? < [\d\s()*+]+$', t):
        return '(by decide)'
    m = re.match(r'^(\d) < (.*)$', t)
    if m:
        q = xmodel.strip_parens(m.group(2))
        q = xmodel.strip_parens(re.sub(r'^\((.*)\) : Nat$', r'\1', q))
        if m.group(1) == '1':
            return xsem.Certs(C.module).gt(q)
        return '(by decide)'
    if ' = ' in t:
        return 'rfl'
    return None


def free_binders(X, texts):
    """Implicit bridge variables not instantiated that `texts` mention (closed under type
    dependencies), with substituted types, in bridge order."""
    imp = [(n, ty) for n, ty in X.implicits() if n not in X.mp]
    names = {n for n, _ in imp}
    need = set()
    for t in texts:
        need |= idents(t) & names
    changed = True
    while changed:
        changed = False
        for n, ty in imp:
            if n in need:
                for d in idents(subst(ty, X.mp)) & names:
                    if d not in need:
                        need.add(d)
                        changed = True
    return [(n, subst(ty, X.mp)) for n, ty in imp if n in need]


def parse_binders(text):
    """`(a b : T) c (d : U)` -> [(name, type or None)]."""
    out, i = [], 0
    text = text.strip()
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text[i] == '(':
            j = xmodel.matching(text, i)
            names, ty = text[i + 1:j].split(' : ', 1)
            out += [(nm, ty.strip()) for nm in names.split()]
            i = j + 1
        else:
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            out.append((text[i:j], None))
            i = j
    return out


def beta(text):
    """Textual beta reduction of applied lambdas `(fun (a : T) b ↦ body) x y` (arguments: a
    parenthesized group or an identifier), repeated to a fixed point."""
    while True:
        done = True
        i = text.find('(fun ')
        while i >= 0:
            j = xmodel.matching(text, i)
            lam = text[i + 1:j]
            m = re.match(r'^fun (.*?) ↦ (.*)$', lam, re.S)
            pre = text[:i].rstrip()
            head = not pre or pre[-1] in '(,' or pre.endswith('↦') or pre.endswith(':=')
            if m and head:
                names = [nm for nm, _ in parse_binders(m.group(1))]
                body = m.group(2)
                k, argv = j + 1, []
                while len(argv) < len(names):
                    while k < len(text) and text[k] == ' ':
                        k += 1
                    if k < len(text) and text[k] == '(':
                        e = xmodel.matching(text, k)
                        argv.append(text[k:e + 1])
                        k = e + 1
                    else:
                        mm = re.match(r"[^\W\d][\w'.]*", text[k:])
                        if not mm or mm.group(0) in ('then', 'else', 'with'):
                            break
                        argv.append(mm.group(0))
                        k += len(mm.group(0))
                if len(argv) == len(names) and names:
                    new = '(' + subst(body, dict(zip(names, argv))) + ')'
                    text = text[:i] + new + text[k:]
                    done = False
                    break
            i = text.find('(fun ', i + 1)
        if done:
            return text


def split_rel(t):
    """`∀ bs, P args → concl` -> (binders, P, args, concl), or None."""
    m = re.match(r'^∀ (.*?), (\w+) (.*)$', t)
    if not m:
        return None
    parts = xmodel.split_top(m.group(3), ' → ')
    if len(parts) != 2:
        return None
    return parse_binders(m.group(1)), m.group(2), xmodel.split_top(parts[0].strip(), ' '), parts[1].strip()


def rel_hyp(C, X, ty):
    """Discharge a relation hypothesis `∀ bs, P args → lhs = rhs` whose `P` is a relation parameter
    of the bundle bound (by the adapter) to a scope `S` with bundle `Y.Comp_v`:
    * a value table `… → v = D j` (`D` a data parameter): `D := fun j ↦ <Y.rel_v's value>`,
      proof `fun … h ↦ Y.rel_v (S.sem h)`;
    * a step `… → nxt = cur + F a j`: `F := fun a j ↦ <Y.hstep_v's term>`, proof `Y.hstep_v (… S.sem …)`.
    Returns (proof, {data parameter: term}, lemma) or None."""
    t = ' '.join(subst(ty, {k: v for k, v in X.mp.items() if k not in X.funs}).split())
    r = split_rel(' '.join(ty.split()))
    rs = split_rel(t)
    if not r or not rs or r[1] not in X.funs:
        return None
    binders, P, args, concl = r
    sb = parse_binders(re.match(r'^∀ (.*?), ', t).group(1))
    m = re.search(r'↦ (\w+) backend', X.funs[P])
    if not m or m.group(1) not in C.ad:
        return None
    S = m.group(1)
    base, v = bridge_names(C.ad[S][0])
    lhs, rhs = [x.strip() for x in xmodel.split_top(concl, ' = ')]
    hyps = {nm for nm, _ in X.hyps()}
    btype = {nm: (subst(ty_, X.mp) if ty_ else 'Nat') for nm, ty_ in binders}
    sidx, _ = index_terms(C, S)
    head, rargs = xmodel.app_split(rhs)
    if head in hyps and head not in X.mp and all(a in btype for a in rargs):
        # value table
        rel = C.B.get(f'{base}.rel{v}')
        hrel = [b for b in rel['binders'] if b[0] == '(' and b[1] == ['h']][0]
        _, rconsts, rmid, _, _, rout = comp_call(hrel[2])
        ymp = const_map(rconsts, C.ad[S][1], binder_types(rel['binders']))
        ridx = [b for b in rmid if not b.startswith('(fun')]
        ymp[ridx[-1]] = f'(({rargs[0]} : Nat))'
        rl, rr = [x.strip() for x in xmodel.split_top(rel['concl'], ' = ')]
        val = subst(rr, ymp)
        term = '(fun ' + ' '.join(f'({a} : {btype[a]})' for a in rargs) + f' ↦ {val})'
        nb = len(binders)
        proof = f'(fun {" ".join(["_"] * nb)} h ↦ {base}.rel{v} ({S}.sem h))'
        return proof, {head: term}, f'{base}.rel{v}'
    mstep = re.match(r'^(\w+) \+ (.*)$', rhs)
    if mstep:
        fpart = mstep.group(2)
        head, fargs = xmodel.app_split(fpart)
        if head not in hyps or head in X.mp:
            return None
        H = C.B.get(f'{base}.hstep{v}')
        hc = [t_ for k, ns, t_ in H['binders'] if k == '(' and ns == ['hcomp']][0]
        hcb = re.match(r'^∀ (.*?), ', ' '.join(hc.split()))
        nb = len(parse_binders(hcb.group(1)))
        inner = hc[hc.index(f'{base}.Comp'):]
        _, hconsts, hmid, _, _, _ = comp_call(inner)
        hmp = const_map(hconsts, C.ad[S][1], binder_types(H['binders']))
        hidx = [b for b in hmid if b not in ('backend',)]
        for b, val in zip(reversed(hidx), reversed(sidx)):
            if NAME.match(b) and val is not None:
                hmp[b] = val
        hr = split_rel(' '.join(H['concl'].split()))
        if not hr:
            return None
        hb_, hP, hargs, hconcl = hr
        for (hn, _), (sn, _) in zip(hb_, sb):
            hmp[hn] = sn
        for ha, sa in zip(hargs, rs[2]):
            if ha.startswith('('):
                for hi, si in zip(xmodel.tuple_items(ha), xmodel.tuple_items(sa)):
                    if NAME.match(hi) and hi not in hmp:
                        hmp[hi] = f'({si})'
        hl, hrr = [x.strip() for x in xmodel.split_top(hconcl, ' = ')]
        mh = re.match(r'^(\w+) \+ (.*)$', hrr)
        if not mh:
            return None
        hterm = mh.group(2)
        val = subst(hterm, hmp)
        term = '(fun ' + ' '.join(f'({a} : {btype.get(a, "Nat")})' for a in fargs) + f' ↦ {val})'
        proof = f'({base}.hstep{v} (fun {" ".join(["_"] * nb)} hc ↦ {S}.sem hc))'
        return proof, {head: term}, f'{base}.hstep{v}'
    return None


def child_step(C, child, X, ty):
    """For a member bundle's `KeyStepHyp child … F`: the variable `F`, its term from the child's
    hstep, the hstep proof, the hstep name and the child loop scope."""
    sc = C.scopes[child].to_json()
    loops = [x for x in sc['atoms'] if x.get('loop')]
    if len(loops) != 1:
        raise ValueError(f'{child}: expected one child loop')
    ch = loops[0]['loop']['child']
    ccomp, cconsts = C.ad[ch]
    cbase, cv = bridge_names(ccomp)
    H = C.B.get(f'{cbase}.hstep{cv}')
    hc = [t for k, ns, t in H['binders'] if k == '(' and ns == ['hcomp']][0]
    _, hconsts, _, _, _, _ = comp_call(hc[hc.index(f'{cbase}.Comp'):])
    hmp = const_map(hconsts, cconsts, binder_types(H['binders']))
    _, pargs = xmodel.app_split(' '.join(subst(ty, X.mp).split()))
    _, hargs = xmodel.app_split(' '.join(H['concl'].split()))
    for hv, pv in zip(hargs[:-1], pargs[:-1]):
        if NAME.match(hv):
            hmp.setdefault(hv, pv)
    Fterm = '(' + xmodel.strip_parens(subst(hargs[-1], hmp)) + ')'
    hh = [hn for k, ns, _ in H['binders'] if k == '(' for hn in ns if hn != 'hcomp']
    proof = f'({cbase}.hstep{cv} (fun _ _ _ hc ↦ {ch}.sem hc) ' + ' '.join(f'({hn} := rfl)' for hn in hh) + ')'
    Fvar = xmodel.app_split(' '.join(ty.split()))[1][-1]
    return Fvar, Fterm, proof, f'{cbase}.hstep{cv}', ch


# --------------------------------------------------------------------------- emitters

HEAD = '''theorem {name} {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}} {{i_0 : Nat}}
    [Fact (i_0 < S)] {{ctx : {inst}.Ctx backend hashModel params i_0}} {{outputs}}
    (F : {inst}.Facts ctx outputs)'''
RESERVED = {'F', 'ctx', 'i', 'i_0', 'S', 'outputs', 'backend', 'hashModel', 'params', 'ex', 'h'}


def per_index(t):
    """Does a (substituted) hypothesis depend on the loop index `i`?"""
    return bool(re.search(r' ctx i\)|\(\(i : Nat\)\)', t))


def binders_text(Mk, free, passed, N):
    s = Mk.binders()
    for n, ty in free:
        if n in RESERVED:
            raise ValueError(f'free bridge variable {n} clashes with a generated name')
        s += f'\n    {{{n} : {ty}}}'
    for n, t in passed:
        if n in RESERVED:
            raise ValueError(f'bridge hypothesis {n} clashes with a generated name')
        s += f'\n    ({n} : ∀ i : Fin {N}, {t})' if per_index(t) else f'\n    ({n} : {t})'
    return s


def emit_par(C, a, kind):
    """A fact `∀ i : Fin N, <bridge conclusion>` for the par loop `a` over a bundle scope."""
    L = a['loop']
    child = L['child']
    if child not in C.ad:
        raise KeyError(f'no bundle adapter for {child}')
    comp, consts = C.ad[child]
    base, v = bridge_names(comp)
    bridge = f'{base}.rel{v}'
    ins = [C.arg(x) for x in L['args']]
    outs = [C.arg(x) for x in L['outs']]
    idx, funs = index_terms(C, child)
    X = Inst(C, bridge, consts, ins, outs, idx, funs)
    nats = C.adm[child][1]
    nmap = {nats[0]: 'i_0', nats[1]: '((i : Nat))'} if len(nats) >= 2 else {nats[0]: '((i : Nat))'} if nats else {}
    for b, term in X.fun_terms.items():
        X.mp.setdefault(b, subst(term, nmap))
    Mk = Masks(C)
    # the secret variable of `sec = fun _ _ ↦ reducePoly q n v` is the instance's secret
    for n, ty in X.hyps():
        t = ' '.join(subst(ty, X.mp).split())
        m = re.match(r'^\(\w+\.(w_\d+_\d+) ctx\) = fun _ _ ↦ reducePoly .* (\w+)$', t)
        if m and Mk.M.kind(m.group(1)) == 'sec':
            X.mp.setdefault(m.group(2), C.secrets[Mk.root(m.group(1))][0])
    extra, note = {}, ''
    # data parameters (`(N : Nat)`, `{pA : Nat}`) fixed by an equation `c = (v : Int)` / `(v : Int) = c`
    # with a numeral `c`: instantiated and passed by name
    fixed = []
    datas = [n for n, ty in X.hyps() if ty in ('Nat', 'Int')] + \
        [n for n, ty in X.implicits() if ty in ('Nat', 'Int') and n not in X.mp]
    for n in datas:
        for _, ty2 in X.hyps():
            t2 = ' '.join(subst(ty2, X.mp).split())
            v = re.escape(f'({n} : Int)')
            m = re.match(r'^\(\((\d+)\) : Int\) = ' + v + '$', t2) or re.match('^' + v + r' = \(\((\d+)\) : Int\)$', t2)
            if m and n not in X.mp:
                X.mp[n] = f'({m.group(1)})'
                fixed.append(f'({n} := {m.group(1)})')
    for n, ty in X.hyps():
        if ty.startswith('KeyStepHyp '):
            Fvar, Fterm, proof, hname, ch = child_step(C, child, X, ty)
            X.mp[Fvar] = Fterm
            extra[n] = proof
            note = f'; child loop `{ch}` by `{hname}`'
            continue
        r = rel_hyp(C, X, ty)
        if r:
            proof, data, used = r
            extra[n] = proof
            for d, term in data.items():
                X.mp[d] = term
                extra[d] = term
            note += f'; `{n}` by `{used}`'
    def run():
        free = X.free() | {nm for nm, _ in X.hyps() if nm not in X.mp}
        args, passed = [], []
        for n, ty in X.hyps():
            if n in extra:
                args.append(f'({n} := {extra[n]})')
                continue
            if n in X.mp:  # a data parameter fixed above
                continue
            t = ' '.join(subst(ty, X.mp).split())
            p = discharge(C, Mk, free, t)
            if p is None:
                passed.append((n, t))
                args.append(f'({n} := {n} i)' if per_index(t) else f'({n} := {n})')
            else:
                args.append(f'({n} := {p})')
        return args, passed
    args, passed = run()
    # gadget matrices and their rebases fixed by the backend: `gadgetOf` / `rebasedGadget`
    gad = gadget_solve(C, X, extra, passed)
    gad = param_solve(C, X, extra) or gad
    if gad:
        args, passed = run()
    concl = beta(' '.join(subst(X.th['concl'], X.mp).split()))
    fb = free_binders(X, [concl] + [t for _, t in passed])
    N = L['count']
    name = f'{kind}_{a["field"]}'
    out = f'/-- `{C.roles.get(child, {}).get("role")}` par[{N}] `{child}` by `{bridge}`{note}. -/\n'
    hb = '\n    (hb : backend = Backend.backend)' if gad else ''
    # per-member form when hypotheses are about the member (their semantic variables are the member's)
    member = any(per_index(t) for _, t in passed)
    if member:
        args = [re.sub(r' i\)$', ')', x) for x in args]
        out += HEAD.format(name=name, inst=C.inst) + f'\n    (i : Fin {N})' + hb + \
            binders_text(Mk, fb, passed, N).replace(f'∀ i : Fin {N}, ', '')
        out += f' :\n    {concl} := by\n'
    else:
        out += HEAD.format(name=name, inst=C.inst) + hb + binders_text(Mk, fb, passed, N)
        out += f' :\n    ∀ i : Fin {N}, {concl} := by\n'
    out += '  subst hb\n' if gad else ''
    out += ''.join(h + '\n' for h in Mk.M.haves)
    lits = [f'({n} := {X.mp[n]})' for n, ty in X.implicits()
            if ty == 'Int' and n in X.mp and re.match(r'^\(\(\(?-?\d+\)?\) : Int\)$', X.mp[n])]
    fv = ' '.join([f'({n} := {n})' for n, _ in fb] + fixed + getattr(X, 'named', []) + lits)
    out += ('' if member else '  intro i\n') + f'  exact {bridge} (h := {child}.sem ({C.I.F(a)} i)) {fv} {" ".join(args)}\n'
    return out, dict(name=name, Mk=Mk, free=fb, passed=passed, concl=concl, N=N, gad=gad)


def param_solve(C, X, extra):
    """Data parameters fixed by instance certificates: a layout `L : RegularLayout q` with
    `backend.regularLayout q n = some L` and `L.droppedModuli = 0` (`Gen_InstCerts` layout cert),
    a cofactor `p` with `Q = p * q'` (`p := Q / q'`, `div` / `quot` certs), an exponent `log` with
    `n = 2 ^ log`. Returns whether a backend certificate was used (the theorem then takes `hb`)."""
    used = False
    hyps = X.hyps()
    tys = {nm: ' '.join(subst(ty, X.mp).split()) for nm, ty in hyps}
    Cc = xsem.Certs(C.module)
    for nm, ty in hyps:
        t = tys[nm]
        m = re.match(r'^RegularLayout (.*)$', t)
        if m and nm not in X.mp:
            q = xmodel.strip_parens(m.group(1))
            qn, inst = xsem.qname(q)
            cert = f'(RefreshTopProof.GenInst.Certs.{C.module}.{qn}_layout{" i_0" if inst else ""})'
            hl = [h for h, ht in tys.items() if re.match(r'^backend\.regularLayout .* = some ' + re.escape(nm) + '$', ht)]
            hx = [h for h, ht in tys.items() if ht == f'{nm}.droppedModuli = 0']
            if not hl or not hx:
                continue
            X.mp[nm] = f'({cert}.choose)'
            extra[nm] = f'{cert}.choose'
            extra[hl[0]] = f'{cert}.choose_spec.1'
            extra[hx[0]] = f'{cert}.choose_spec.2.1'
            used = True
    for nm, ty in X.implicits():
        if nm in X.mp or ty != 'Nat':
            continue
        for h, t in tys.items():
            m = re.match(r'^\(?(\S+?)\)? = ' + re.escape(nm) + r' \* (.*)$', t)
            if m and h not in extra:
                Q, q2 = xmodel.strip_parens(m.group(1)), xmodel.strip_parens(m.group(2))
                hq, hp = Cc.div(Q, q2)
                X.mp[nm] = f'({xsem.qterm(Q)} / {xsem.qterm(q2)})'
                X.named = getattr(X, 'named', []) + [f'({nm} := {X.mp[nm]})']
                extra[h] = hq
                for h2, t2 in tys.items():
                    if t2 == f'0 < {nm}':
                        extra[h2] = f'(lt_trans Nat.zero_lt_one {hp})'
                    elif t2 == f'1 < {nm}':
                        extra[h2] = hp
                break
            m = re.match(r'^\(?(\d+)\)? = 2 \^ ' + re.escape(nm) + '$', t)
            if m and h not in extra:
                v = int(m.group(1))
                if v & (v - 1) == 0:
                    X.mp[nm] = f'({v.bit_length() - 1})'
                    X.named = getattr(X, 'named', []) + [f'({nm} := {X.mp[nm]})']
                    extra[h] = 'rfl'
                break
    return used


def gadget_solve(C, X, extra, passed):
    """Instantiate free gadget variables `V` with `gadgetMatrixRuns backend B D V` by
    `gadgetOf backend B D q n m` (proof from a backend layout certificate) and free `Y` with
    `centeredRebaseRuns V Y` by `rebasedGadget …`. Returns whether any was solved."""
    free = X.free()
    imp = dict(X.implicits())
    solved = {}
    for n, t in passed:
        head, args = xmodel.app_split(t)
        if head == 'gadgetMatrixRuns' and len(args) == 4 and args[0] == 'backend' and args[3] in free:
            V = args[3]
            _, q, nn, r, m = xsem.mat_type(subst(imp[V], X.mp))
            if r != '1':
                continue
            B, D = args[1], args[2]
            cert = C.gadget_cert(q, nn, m, B, D)
            X.mp[V] = f'(gadgetOf backend {B} {D} {xsem.qterm(q)} {nn} {m})'
            extra[n] = f'(gadgetOf_runs {cert})'
            solved[V] = (B, D, q, nn, m)
    for n, t in passed:
        head, args = xmodel.app_split(t)
        if head == 'centeredRebaseRuns' and len(args) == 2 and args[0] in solved and args[1] in free:
            Y = args[1]
            B, D, q, nn, m = solved[args[0]]
            _, q2, _, _, _ = xsem.mat_type(subst(imp[Y], X.mp))
            X.mp[Y] = f'(rebasedGadget backend {B} {D} {xsem.qterm(q)} {xsem.qterm(q2)} {nn} {m})'
            extra[n] = f'(rebasedGadget_runs {xsem.Certs(C.module).gt(q2)})'
    return bool(solved)


def rhs_of(C, k):
    """`rhs_<field>`: all cells' right-hand sides with one error family (online `hrhs` / `heB`)."""
    concl, N = k['concl'], k['N']
    body, pre = concl, 0
    while body.startswith('∃ ') and not re.match(r'^∃ eB\b', body):
        body = body[body.index(', ') + 2:]
        pre += 1
    parts = xmodel.split_top(body, ' ∧ ')
    ke = [j for j, p in enumerate(parts) if p.strip().startswith('∃ ')][0]
    ex = ' ∧ '.join(parts[ke:]).strip()
    parts = parts[:ke + 1]
    m = re.match(r'^∃ (\w+) : (.*?), \(∀ j, (.*)\) ∧ ∀ j, (.*)$', ex)
    if not m:
        raise ValueError('rhs: no error family conjunct')
    e, T, bnd, eq = m.groups()
    ei = lambda s: re.sub(r'\b' + e + r' j\b', f'{e} i j', s)
    # the scale `((pp : Int) : ErrorPoly n)` of the plaintext `pp · message c e` as `((p' : Nat) : Int)`
    # (the online cells' `(p' : Int) * ν`): `p'` with `hp' : (p' : Int) = pp`
    scale = None
    k0 = eq.find(' * message ')
    if k0 > 0 and eq[k0 - 1] == ')':
        depth, j = 0, k0 - 1
        while j >= 0:
            depth += {')': 1, '(': -1}.get(eq[j], 0)
            if depth == 0:
                break
            j -= 1
        st = eq[j:k0]
        mm = re.match(r'^\(\((.*)\) : ErrorPoly (.*)\)$', st)
        if mm and mm.group(1).endswith(' : Int'):
            scale = (mm.group(1)[:-len(' : Int')].strip(), mm.group(2))
            eq = eq[:j] + f"(((p' : Nat) : Int) : ErrorPoly {mm.group(2)})" + eq[k0:]
    name = 'rhs_' + k['name'].split('_', 1)[1]
    pat = '⟨' + ', '.join(['_'] * pre + ['-'] * (len(parts) - 1) + ['h']) + '⟩'
    call = ' '.join([k['name'], 'F'] + (['hb'] if k['gad'] else []) + [k['Mk'].args()] + [f'({n} := {n})' for n, _ in k['free']] +
                    [n for n, _ in k['passed']])
    out = f'/-- The right-hand sides of all cells of `{k["name"]}` with one error family (the online cell gather\'s `hrhs` / `heB`). -/\n'
    out += HEAD.format(name=name, inst=C.inst) + ('\n    (hb : backend = Backend.backend)' if k['gad'] else '') + \
        binders_text(k['Mk'], k['free'], k['passed'], N)
    exn = ex
    if scale:
        out += f"\n    {{p' : Nat}} (hp' : ((p' : Nat) : Int) = {scale[0]})"
        exn = ex.replace(ex[ex.index(', ') + 2:], f'(∀ j, {bnd}) ∧ ∀ j, {eq}')
    out += (f' :\n    ∃ {e} : Fin {N} → {T}, (∀ (i : Fin {N}) j, {ei(bnd)}) ∧\n'
            f'      ∀ (i : Fin {N}) j, {ei(eq)} := by\n')
    out += f'  have ex : ∀ i : Fin {N}, {exn} := fun i ↦ by\n'
    out += f'    obtain {pat} := {call} i\n'
    out += ("    rw [← hp'] at h\n" if scale else '') + '    exact h\n'
    out += '  exact ⟨fun i ↦ (ex i).choose, fun i ↦ (ex i).choose_spec.1, fun i ↦ (ex i).choose_spec.2⟩\n'
    return out, name


def emit_keyloop(C, a):
    """A root key loop from zero keys: the final keys are the sum of the branch keys."""
    L = a['loop']
    ch = L['child']
    ccomp, cconsts = C.ad[ch]
    cbase, cv = bridge_names(ccomp)
    H = C.B.get(f'{cbase}.hstep{cv}')
    hc = [t for k, ns, t in H['binders'] if k == '(' and ns == ['hcomp']][0]
    _, hconsts, _, _, hins, _ = comp_call(hc[hc.index(f'{cbase}.Comp'):])
    hmp = const_map(hconsts, cconsts, binder_types(H['binders']))
    for p, v in zip(xmodel.tuple_items(hins), L['args']):
        if NAME.match(p) and xsem.wire_of(v.split()[0]):
            hmp.setdefault(p, C.arg(v))
    _, hargs = xmodel.app_split(' '.join(H['concl'].split()))
    Fterm = xmodel.strip_parens(subst(hargs[-1], hmp))
    fin = '(' + ', '.join(C.W(w) for w in xsem.tuple_wires(L['final'])) + ', ())'
    hh = [hn for k, ns, _ in H['binders'] if k == '(' for hn in ns if hn != 'hcomp']
    name = f'keyloop_{a["field"]}'
    out = f'/-- `{C.roles.get(ch, {}).get("role")}` seq[{L["count"]}] `{ch}` from zero keys (`key_loop_of`). -/\n'
    out += HEAD.format(name=name, inst=C.inst)
    out += f' :\n    {fin} = ∑ k ∈ Finset.range {L["count"]}, ({Fterm}) k :=\n'
    C.keyloops[name] = (xsem.tuple_wires(L['final']), f'∑ k ∈ Finset.range {L["count"]}, ({Fterm}) k')
    out += (f'  BrLutKey.key_loop_of ({cbase}.hstep{cv} (fun _ _ _ hc ↦ {ch}.sem hc) '
            + ' '.join(f'({hn} := rfl)' for hn in hh) + f') {C.I.F(a)}\n')
    return out, name


def emit_preimages(C):
    out = []
    for a in C.ij['atoms']:
        if a.get('loop') or not a['head'].endswith('preimageRunsDispatched'):
            continue
        A, T, tgt, cut, P = [x.strip() for x in a['args'][1:]]
        A, T, tgt, P = map(xsem.wire_of, (A, T, tgt, P))
        td = [b for b in C.ij['atoms'] if not b.get('loop') and b['head'].endswith('trapdoorSample')
              and [xsem.wire_of(x) for x in b['args'][-2:]] == [A, T]]
        if not td:
            continue
        _, q, nn, inner, cols = xsem.mat_type(C.ij['wires'][P]['type'])
        name = 'preimage_' + a['field']
        out.append((f'''/-- The decoder preimage: `A P = target` with `P` small (`preimage_target_gen`). -/
{HEAD.format(name=name, inst=C.inst)} :
    MxxRuntime.matrixMul {C.W(A)} {C.W(P)} = {C.W(tgt)} ∧ ∃ d : ErrorMatrix {nn} {inner} {cols},
      {C.W(P)} = reduceMatrix {xsem.qterm(q)} {nn} {inner} {cols} d ∧ CoeffBound d {cut} :=
  preimage_target_gen {C.I.F(td[0])} {C.I.F(a)}
''', name))
    return out


def emit_links(C):
    """Equal hash samples of the instance body (same model, prefix, tags and key)."""
    seen, out = {}, []
    for a in C.ij['atoms']:
        if a.get('loop') or not a['head'].endswith('hashSample'):
            continue
        key = tuple(' '.join(x.split()) for x in a['args'][:-1])
        w = xsem.wire_of(a['args'][-1])
        if key in seen:
            b, wb = seen[key]
            name = f'link_{wb}_{w}'
            out.append((f'''/-- `{wb}` and `{w}` are hash samples under the same tags and key. -/
{HEAD.format(name=name, inst=C.inst)} :
    {C.W(wb)} = {C.W(w)} :=
  Gen.hashSample_unique {C.I.F(b)} {C.I.F(a)}
''', name))
        else:
            seen[key] = (a, w)
    return out



# --------------------------------------------------------------------------- root private key cells

def _mod_facts(C, Mk, w, haves):
    """(kind, proof) for `W w = fun _ _ ↦ reducePoly q n s` ('sec') or `W w = 1` ('one'), through
    centered rebases (Masks) and modulus reductions."""
    k = Mk.M.kind(w)
    if k is not None:
        f = Mk.M.fact(w)
        if f is not None:
            if k == 'sec':
                Mk.used.add(Mk.root(w))
            return k, f
    a = C.I.prod.get(w)
    if not a or not a['head'].endswith('modulusReduceRuns'):
        return None, None
    src = xsem.wire_of(a['args'][0])
    ks, fs = _mod_facts(C, Mk, src, haves)
    if ks is None:
        return None, None
    qs = xsem.mat_type(C.ij['wires'][src]['type'])[1]
    qd = xsem.mat_type(C.ij['wires'][w]['type'])[1]
    cert = xsem.Certs(C.module)
    gs, gd = cert.gt(qs), cert.gt(qd)
    name = f'r_{w}'
    nn = xsem.mat_type(C.ij['wires'][w]['type'])[2]
    if ks == 'one':
        if fs == 'rfl':
            haves.append(f'  have {name} : {C.W(w)} = 1 := modulusReduce_one (q := {xsem.qterm(qs)}) {gs} {gd} (by decide) {C.I.F(a)}')
        else:
            haves.append(f'  have {name} : {C.W(w)} = 1 := modulusReduce_one (q := {xsem.qterm(qs)}) {gs} {gd} (by decide) (by have h := {C.I.F(a)}; rw [{fs}] at h; exact h)')
    else:
        sv = C.secrets[Mk.root(src) if src not in C.secrets else src][0] if True else 'sp'
        haves.append(f'  have {name} : {C.W(w)} = fun _ _ ↦ reducePoly {xsem.qterm(qd)} {nn} {sv} := '
                     f'modulusReduce_poly (q := {xsem.qterm(qs)}) {gs} {gd} (by decide) (by have h := {C.I.F(a)}; rw [{fs}] at h; exact h)')
    return ks, name


ROOTCELL_DOC = ('/-- Root private key cell `{c0}`/`{c1}`: `s • t_j + c_j = (pc μ (s, 1)_j) G + e_j` '
                '(`root_cell`; tops `topPairG`, payload `message c e`). -/\n')


def emit_rootcells(C):
    """Root private key cells `c_j = (σ_j G) [pc] μ - s t_j + e_j` (σ_0 = s, σ_1 = 1) of the instance body,
    paired by gadget / payload / scale / tag."""
    L = C.I.lets

    def let(w, head):
        tm = L.get(w, '')
        m = re.match(r'^MxxRuntime\.' + head + r' (w_\d+_\d+) (w_\d+_\d+)$', tm)
        return m.groups() if m else None

    cells = []
    for c, tm in L.items():
        m = re.match(r'^MxxRuntime\.matrixAdd (w_\d+_\d+) (w_\d+_\d+)$', tm)
        if not m:
            continue
        A, E = m.groups()
        ea = C.I.prod.get(E)
        if not ea or not ea['head'].endswith('gaussianSample'):
            continue
        s1 = let(A, 'matrixSub')
        if not s1:
            continue
        X, Y = s1
        y = let(Y, 'matrixMulScalarLeft')
        x = let(X, 'matrixMulScalarRight')
        if not (x and y):
            continue
        Z, pay = x
        z = let(Z, 'matrixMulScalarRight')
        if not z:
            continue
        V, PC = z
        mp = re.match(r'^\(MxxRuntime\.matrixPolynomial \[\((.*?)\)\] : .*\)$', L.get(PC, ''))
        v = let(V, 'matrixMulScalarLeft')
        if not (mp and v):
            continue
        o, G = v
        s, top = y
        ta = C.I.prod.get(top)
        if not ta or not ta['head'].endswith('hashSample'):
            continue
        cells.append(dict(c=c, e=E, ea=ea, s=s, o=o, G=G, pay=pay, pc=mp.group(1), top=top, ta=ta))
    out = []
    used = set()
    for a in cells:
        if a['o'] != a['s'] or a['c'] in used:
            continue
        b = [x for x in cells if x['o'] != x['s'] and x['s'] == a['s'] and x['G'] == a['G'] and x['pay'] == a['pay']
             and x['pc'] == a['pc'] and x['ta']['args'][:2] == a['ta']['args'][:2] and x['ta']['args'][3] == a['ta']['args'][3]]
        if len(b) != 1:
            continue
        b = b[0]
        used |= {a['c'], b['c']}
        Mk = Masks(C)
        haves = []
        ks, fs = _mod_facts(C, Mk, a['s'], haves)
        ko, fo = _mod_facts(C, Mk, b['o'], haves)
        if ks != 'sec' or ko != 'one':
            continue
        pa = C.I.prod.get(a['pay'])
        if not pa or not pa['head'].endswith('modulusReduceRuns'):
            continue
        psrc = xsem.wire_of(pa['args'][0])
        mm = re.match(r'^MxxRuntime\.multiplyMonomial (w_\d+_\d+) (w_\d+_\d+)$', L.get(psrc, ''))
        if not mm:
            continue
        lw, ew = mm.groups()
        ml = re.match(r'^MxxRuntime\.liftInteger (w_\d+_\d+)$', L.get(lw, ''))
        if not ml:
            continue
        cw = ml.group(1)
        qs = xsem.mat_type(C.ij['wires'][psrc]['type'])[1]
        _, q, nn, _, m = xsem.mat_type(C.ij['wires'][a['c']]['type'])
        q = xsem.qterm(q)
        cert = xsem.Certs(C.module)
        haves.append(f'  have hpay : {C.W(a["pay"])} = fun _ _ ↦ reducePoly {q} {nn} (message {C.W(cw)} {C.W(ew)}) :=\n'
                     f'    modulusReduce_poly (q := {xsem.qterm(qs)}) {cert.gt(qs)} {cert.gt(q)} (by decide) (by have h := {C.I.F(pa)}; '
                     f'rw [show {C.W(psrc)} = _ from multiplyMonomial_liftInteger _ _] at h; exact h)')
        model = a['ta']['args'][0].strip()
        tag = a['ta']['args'][1].strip()
        comps = xmodel.strip_parens(a['ta']['args'][2].strip())
        items = xmodel.split_top(comps[1:-1], ',')
        pre = '[' + ', '.join(x.strip() for x in items[:-2]) + ']'
        key = a['ta']['args'][3].strip()
        kw = xsem.wire_of(key)
        if kw:
            key = C.W(kw)
        cutv = re.search(r'gaussianSample \S+ \(?(-?\d+)\)? ', C.fields[a['ea']['field']][1])
        name = f'rootcell_{a["c"]}'
        body = '\n'.join(Mk.M.haves + haves)
        concl = (f'∃ e : Fin 2 → ErrorMatrix {nn} 1 {m}, (∀ j, CoeffBound (e j) (({cutv.group(1)}) : Int).toNat) ∧ ∀ j,\n'
                 f'      reducePoly {q} {nn} sp • topPairG {model} {key} {q} {m} {tag} {pre} j + ![{C.W(a["c"])}, {C.W(b["c"])}] j =\n'
                 f'        reducePoly {q} {nn} ((({a["pc"]} : Int) : ErrorPoly {nn}) * message {C.W(cw)} {C.W(ew)} * unitPair sp j) • {C.W(a["G"])} +\n'
                 f'          reduceMatrix {q} {nn} 1 {m} (e j)')
        txt = ROOTCELL_DOC.format(c0=a['c'], c1=b['c'])
        txt += HEAD.format(name=name, inst=C.inst) + Mk.binders() + ' :\n    ' + concl + ' := by\n' + body + '\n'
        txt += (f'  have ht : ![{C.W(a["top"])}, {C.W(b["top"])}] = topPairG {model} {key} {q} {m} {tag} {pre} := by\n'
                f'    funext j\n    fin_cases j\n'
                f'    · exact topPairG_of_hash (pre := {pre}) 0 rfl {C.I.F(a["ta"])}\n'
                f'    · exact topPairG_of_hash (pre := {pre}) 1 rfl {C.I.F(b["ta"])}\n'
                f'  rw [← ht]\n'
                f'  obtain ⟨e, hb, he⟩ := root_cell {fs} {fo} {C.W(a["G"])} {C.W(a["top"])} {C.W(b["top"])} ({a["pc"]}) hpay '
                f'{C.I.F(a["ea"])} {C.I.F(b["ea"])}\n'
                f'  exact ⟨e, hb, fun j ↦ by fin_cases j <;> exact he _⟩\n')
        out.append((txt, name))
    return out


# --------------------------------------------------------------------------- structural loops and the preimage target

def resolve_inputs(text, items):
    """Replace `inputs.2.2.1`-style projections of the loop input tuple by the items."""
    def rep(m):
        path = m.group(1) or ''
        steps = re.findall(r'\.([12])', path)
        k = 0
        for j, st in enumerate(steps):
            if st == '2':
                k += 1
            else:
                if j != len(steps) - 1:
                    raise ValueError(f'projection {path}')
                return items[k]
        return '(' + ', '.join(items[k:]) + ', ())'
    return re.sub(r'(?<![\w.])inputs((?:\.[12])*)(?![\w.])', rep, text)


def named(args, key):
    for x in args:
        m = re.match(r'^\(' + re.escape(key) + r' := (.*)\)$', x)
        if m:
            return xmodel.strip_parens(m.group(1))
    raise KeyError(key)


def positional(args):
    return [x for x in args if not re.match(r'^\(\w+ := ', x)]


def conj_proj(c, k):
    """Projection of component `c` of a `k`-fold conjunction."""
    return '.2' * c + ('.1' if c < k - 1 else '')


def loop_items(C, L):
    return [C.arg(x) if xsem.wire_of(x.split()[0]) else x for x in L['args']]


def elem_type(C, w):
    _, q, nn, r, m = xsem.mat_type(C.ij['wires'][w]['type'])
    return xsem.qterm(q), nn, r, m


def wire_term(t):
    m = re.match(r'^\(?\w+\.(w_\d+_\d+) ctx\)?$', xmodel.strip_parens(t))
    if not m:
        raise ValueError(f'not a wire term: {t}')
    return m.group(1)


def emit_struct_par(C, a):
    """`spec_<field>`: the role fact of every member of a par loop over a structural scope."""
    L = a['loop']
    child = L['child']
    concl, nats, hm = C.sad[child]
    items = [C.arg(x) for x in L['args']]
    outs = [C.arg(x) for x in L['outs']]
    tin = C.scopes[child].input_type or ''
    ins = items[0] if len(items) == 1 and ' × ' not in tin else '(' + ', '.join(items) + ', ())'
    tout = C.scopes[child].output_type or ''
    outt = outs[0] if len(outs) == 1 and ' × ' not in tout else '(' + ', '.join(outs) + ', ())'
    mp = {nats[-1]: '((i : Nat))', hm: 'hashModel', 'inputs': ins, 'outputs': outt}
    if len(nats) >= 2:
        mp[nats[0]] = 'i_0'
    st = ' '.join(subst(concl, mp).split())
    name = f'spec_{a["field"]}'
    N = L['count']
    out = f'/-- `{C.roles.get(child, {}).get("role")}` par[{N}] `{child}`: its role fact for every member. -/\n'
    out += HEAD.format(name=name, inst=C.inst) + f' :\n    ∀ i : Fin {N}, {st} :=\n'
    out += f'  fun i ↦ {child}.sem ({C.I.F(a)} i)\n'
    return out, name


class Seq:
    """A seq loop over a structural step scope: the role spec's arguments resolved at the instance."""

    def __init__(self, C, a):
        self.a, self.L = a, a['loop']
        self.child = self.L['child']
        concl, nats, hm = C.sad[self.child]
        self.nats = nats
        self.head, self.args = xmodel.app_split(resolve_inputs(concl, loop_items(C, self.L)))
        self.pos = positional(self.args)
        self.init = xsem.tuple_wires(self.L['init'])
        self.final = xsem.tuple_wires(self.L['final'])
        self.count = self.L['count']


def get_static(C, w, idx):
    """The family of the `familyGetStatic fam (idx) w` producing `w`, and the atom."""
    a = C.I.prod.get(w)
    if not a or not a['head'].endswith('familyGetStatic') or a['args'][1].strip('()') != str(idx):
        raise ValueError(f'{w}: not a static read of member {idx}')
    return xsem.wire_of(a['args'][0]), a


def emit_memsum(C, a):
    """`memsum_<field>`: a member-sum loop started from members 0 and 1 (iteration `i` adds member
    `2 + i`) is, per component, the sum of its first `count + 2` members."""
    Q = Seq(C, a)
    if Q.head != 'MemberStep.Spec':
        raise ValueError('not a member step')
    k = int(named(Q.args, 'k'))
    fams_t, idx_t = Q.pos[0], Q.pos[1]
    fw = [wire_term(x) for x in xmodel.tuple_items(fams_t)]
    if not re.match(r'^2 \+ \(?Int\.ofNat ' + Q.nats[-1] + r'\)?$', xmodel.strip_parens(idx_t)):
        raise ValueError(f'member index {idx_t}')
    q, nn, r, m = elem_type(C, fw[0])
    eqs, pfs = [], []
    for c in range(k):
        h, xs = xsem.app(C.I.lets.get(Q.init[c], ''))
        if h != 'matrixAdd':
            raise ValueError(f'init {Q.init[c]} not a sum of two reads')
        f0, a0 = get_static(C, xsem.wire_of(xs[0]), 0)
        f1, a1 = get_static(C, xsem.wire_of(xs[1]), 1)
        if f0 != fw[c] or f1 != fw[c]:
            raise ValueError('init reads another family')
        eqs.append(f'{C.W(Q.final[c])} = ∑ i ∈ Finset.range {Q.count + 2}, famAt {C.W(fw[c])} i')
        pfs.append(f'memsum_full {C.I.F(a0)} {C.I.F(a1)} rfl (hm.2 {c})')
    name = f'memsum_{a["field"]}'
    out = f'/-- `{C.roles.get(Q.child, {}).get("role")}` seq[{Q.count}] `{Q.child}`: the sums of the first `count + 2` members. -/\n'
    out += HEAD.format(name=name, inst=C.inst) + ' :\n    ' + ' ∧\n    '.join(eqs) + ' := by\n'
    out += f'  have hr := steps_of {C.I.F(a)} (fun _ _ _ hb ↦ {Q.child}.sem hb)\n'
    out += (f'  have hm := MemberSum.sem (α := ExactMatrix {q} {nn} {r} {m}) (k := {k}) (fams := {fams_t}) '
            f'(idx := fun {Q.nats[-1]} ↦ {idx_t}) hr (fun _ _ _ h ↦ h)\n')
    out += '  exact ⟨' + ', '.join(pfs) + '⟩\n'
    return out, name, dict(final=Q.final, fams=fw, count=Q.count, k=k)


def emit_colsum(C, a, memsums):
    """`colsum_<field>`: a `Step::Sum` column loop whose initial blocks are the member sums of
    column 0 holds, per component, the sum of members `W i + j` (`j < W`) at column `i`."""
    Q = Seq(C, a)
    if Q.head != 'ColStep.Spec':
        raise ValueError('not a column step')
    k, Cc, m = int(named(Q.args, 'k')), named(Q.args, 'C'), named(Q.args, 'm')
    fams_t, Wt = Q.pos[0], Q.pos[1]
    fw = [wire_term(x) for x in xmodel.tuple_items(fams_t)]
    q, nn, _, _ = elem_type(C, fw[0])
    eqs, pfs = [], []
    for c in range(k):
        ta = C.I.prod.get(Q.init[c])
        if not ta or not ta['head'].endswith('tensorRuns'):
            raise ValueError(f'init {Q.init[c]} not a tensor')
        u, col0 = xsem.wire_of(ta['args'][0]), xsem.wire_of(ta['args'][1])
        if not re.match(r'^\(MxxRuntime\.unitRow \(0\) :', C.I.lets.get(u, '')):
            raise ValueError('initial block not at column 0')
        src = [(n, d['final'].index(col0)) for n, d in memsums.items() if col0 in d['final']]
        if not src:
            raise ValueError(f'column 0 {col0} not a member sum')
        mn, mc = src[0]
        d = memsums[mn]
        if d['fams'][mc] != fw[c] or d['count'] + 2 != int(Wt):
            raise ValueError('column 0 sums other members')
        eqs.append(f'{C.W(Q.final[c])} = ∑ i ∈ Finset.range {Q.count + 1}, colPlace {q} {nn} {Cc} {m} i '
                   f'(∑ j ∈ Finset.range {Wt}, famAt {C.W(fw[c])} ({Wt} * i + j))')
        pfs.append(f'colsum_full ({mn} F){conj_proj(mc, d["k"])} {C.I.F(ta)} '
                   f'(colsum_loop (q := {q}) (C := {Cc}) (m := {m}) (k := {k}) (W := {Wt}) (fams := {fams_t}) '
                   f'hr (fun _ _ _ h ↦ h) {c})')
    name = f'colsum_{a["field"]}'
    out = f'/-- `{C.roles.get(Q.child, {}).get("role")}` seq[{Q.count}] `{Q.child}`: column `i` holds the sum of members `W i + j`, `j < W`. -/\n'
    out += HEAD.format(name=name, inst=C.inst) + ' :\n    ' + ' ∧\n    '.join(eqs) + ' := by\n'
    out += f'  have hr := steps_of {C.I.F(a)} (fun _ _ _ hb ↦ {Q.child}.sem hb)\n'
    out += '  exact ⟨' + ', '.join(pfs) + '⟩\n'
    return out, name


def emit_slicesum(C, a):
    """`slicesum_<field>`: the slice form of a column loop with its initial block 0: block `i` of
    `x` times `v`, placed at column `i`, for all `count + 1` blocks."""
    Q = Seq(C, a)
    if Q.head != 'ColSliceStep.Spec':
        raise ValueError('not a slice step')
    Cc, m = named(Q.args, 'C'), named(Q.args, 'm')
    X, V = Q.pos[0], Q.pos[1]
    xw = wire_term(X)
    q, nn, _, _ = elem_type(C, xw)
    ta = C.I.prod.get(Q.init[0])
    if not ta or not ta['head'].endswith('tensorRuns'):
        raise ValueError('init not a tensor')
    u, mv = xsem.wire_of(ta['args'][0]), xsem.wire_of(ta['args'][1])
    if not re.match(r'^\(MxxRuntime\.unitRow \(0\) :', C.I.lets.get(u, '')):
        raise ValueError('initial block not at column 0')
    h, xs = xsem.app(C.I.lets.get(mv, ''))
    if h != 'matrixMul':
        raise ValueError('initial block not a product')
    sa = C.I.prod.get(xsem.wire_of(xs[0]))
    if not sa or not sa['head'].endswith('sliceMatrix') or xsem.wire_of(sa['args'][0]) != xw:
        raise ValueError('initial block not a slice of x')
    name = f'slicesum_{a["field"]}'
    st = (f'{C.W(Q.final[0])} = ∑ i ∈ Finset.range {Q.count + 1}, colPlace {q} {nn} {Cc} 1 i '
          f'(matrixMul (sliceOf {q} {nn} ({Cc} * {m}) {m} {X} (({m} : Nat) * (i : Int)) '
          f'(({m} : Nat) * (i : Int) + ({m} : Nat))) {V})')
    out = f'/-- `{C.roles.get(Q.child, {}).get("role")}` seq[{Q.count}] `{Q.child}`: block `i` of `x` times `v` at column `i`. -/\n'
    out += HEAD.format(name=name, inst=C.inst) + f' :\n    {st} := by\n'
    out += f'  have hr := steps_of {C.I.F(a)} (fun _ _ _ hb ↦ {Q.child}.sem hb)\n'
    out += f'  have hl := slicesum_loop (q := {q}) (C := {Cc}) (m := {m}) (x := {X}) (v := {V}) hr (fun _ _ _ h ↦ h)\n'
    out += f'  exact slicesum_full {C.I.F(sa)} {C.I.F(ta)} hl\n'
    return out, name


def emit_bms(C):
    out = []
    for a in C.ij['atoms']:
        if a.get('loop') or not a['head'].endswith('blockModSwitchRuns'):
            continue
        basis, plain, x, y = [z.strip() for z in a['args']]
        name = f'bms_{a["field"]}'
        txt = f'/-- The block modulus switch `{a["field"]}` as a function. -/\n'
        txt += HEAD.format(name=name, inst=C.inst)
        txt += f' :\n    {C.W(xsem.wire_of(y))} = blockModSwitchG {basis} {plain} {C.W(xsem.wire_of(x))} :=\n'
        txt += f'  blockModSwitchRuns_eqG {C.I.F(a)}\n'
        out.append((txt, name))
    return out


def flatten_add(C, w):
    h, xs = xsem.app(C.I.lets.get(w, ''))
    if h == 'matrixAdd':
        return flatten_add(C, xsem.wire_of(xs[0])) + flatten_add(C, xsem.wire_of(xs[1]))
    return [w]


def emit_target(C):
    """`target_<field>`: each row `k` of the preimage equation `A P = target` is the sum of the
    combination, fresh and smudge keys of the row (`GenFinal`'s `Bk k * P = Kc k + Kf k + Ks k`):
    leaves of the row's sum classified by the role of the loop producing them (slice-form column
    loop: fresh; member column loop: smudge; otherwise combination). `comb_<w>`: a combination key
    split at its concatenated columns."""
    finals = {}
    for a in C.ij['atoms']:
        L = a.get('loop')
        if L and L['kind'] == 'seq':
            for w in xsem.tuple_wires(L['final']):
                finals[w] = C.roles.get(L['child'], {}).get('role')
    out = []
    for a in C.ij['atoms']:
        if a.get('loop') or not a['head'].endswith('preimageRunsDispatched'):
            continue
        A, T, tgt, cut, P = [x.strip() for x in a['args'][1:]]
        A, T, tgt, P = map(xsem.wire_of, (A, T, tgt, P))
        td = [b for b in C.ij['atoms'] if not b.get('loop') and b['head'].endswith('trapdoorSample')
              and [xsem.wire_of(x) for x in b['args'][-2:]] == [A, T]]
        cat = C.I.prod.get(tgt)
        if not td or not cat or not cat['head'].endswith('concatRows'):
            raise ValueError('preimage target not two concatenated rows')
        rows = [xsem.wire_of(x) for x in cat['args'][:2]]
        parts = {'Kc': [], 'Kf': [], 'Ks': []}
        for r in rows:
            leaves = flatten_add(C, r)
            kinds = []
            for w in leaves:
                role = finals.get(w)
                kinds.append('Kf' if role == 'sum.columns.slice-step' else
                             'Ks' if role == 'sum.columns.step' else 'Kc')
            if kinds != ['Kc', 'Kf', 'Ks']:
                raise ValueError(f'row {r}: leaves {list(zip(leaves, kinds))}')
            for w, k in zip(leaves, kinds):
                parts[k].append(w)
        vec = lambda ws: '![' + ', '.join(C.W(w) for w in ws) + ']'
        name = f'target_{a["field"]}'
        txt = ('/-- The preimage target by rows: `rowOf A k * P = Kc k + Kf k + Ks k` (combination, fresh '
               'and smudge keys of row `k`; `GenFinal.final_gen`\'s `hpre`). -/\n')
        txt += HEAD.format(name=name, inst=C.inst)
        txt += (f' :\n    ∀ k : Fin 2, rowOf {C.W(A)} k * {C.W(P)} =\n'
                f'      {vec(parts["Kc"])} k + {vec(parts["Kf"])} k + {vec(parts["Ks"])} k := by\n')
        txt += '  intro k\n'
        txt += f'  have hP := (preimage_target_gen {C.I.F(td[0])} {C.I.F(a)}).1\n'
        txt += f'  obtain ⟨h0, h1⟩ := concatRows_two {C.I.F(cat)}\n'
        txt += '  rw [← rowOf_mul, ← matrixMul, hP]\n'
        txt += '  refine Fin.cases ?_ (fun j ↦ ?_) k\n  · exact h0\n  · rw [Fin.fin_one_eq_zero j]; exact h1\n'
        out.append((txt, name))
        for w in parts['Kc']:
            h, xs = xsem.app(C.I.lets.get(w, ''))
            if h != 'matrixSub':
                continue
            x, y = xsem.wire_of(xs[0]), xsem.wire_of(xs[1])
            hx, xx = xsem.app(C.I.lets.get(x, ''))
            hy, yy = xsem.app(C.I.lets.get(y, ''))
            if hx != 'matrixMul' or hy != 'matrixMul':
                continue
            gc, D = xsem.wire_of(xx[0]), xsem.wire_of(xx[1])
            cc = C.I.prod.get(gc)
            if not cc or not cc['head'].endswith('concatColumns'):
                continue
            g, c1 = xsem.wire_of(cc['args'][0]), xsem.wire_of(cc['args'][1])
            Kx, Y = xsem.wire_of(yy[0]), xsem.wire_of(yy[1])
            cname = f'comb_{w}'
            txt = (f'/-- The combination key `{w}`: the concatenated keys times the decomposition, minus the '
                   'second product (`concatColumns_mul`). -/\n')
            txt += HEAD.format(name=cname, inst=C.inst)
            txt += (f' :\n    {C.W(w)} = {C.W(g)} * topRows {C.W(D)} + {C.W(c1)} * bottomRows {C.W(D)} - '
                    f'{C.W(Kx)} * {C.W(Y)} := by\n')
            txt += f'  show {C.W(gc)} * {C.W(D)} - {C.W(Kx)} * {C.W(Y)} = _\n'
            txt += f'  rw [concatColumns_mul {C.I.F(cc)}]\n'
            out.append((txt, cname))
    return out


def emit_selects(C):
    """`sel_<field>`: a `select (Int.ofNat i_0) [x_0, …] out` is member `i_0` of the list;
    `selkeys_<field>`: the same with every item a component of a root key loop's final keys,
    written through the loop's branch-key sum (`keyloop_<f>`)."""
    out = []
    for a in C.ij['atoms']:
        if a.get('loop') or a['head'].split('.')[-1] != 'select':
            continue
        iw, lst, ow = a['args'][0], a['args'][1], a['args'][-1]
        if not re.match(r'^\(?Int\.ofNat i_0\)?$', xmodel.strip_parens(C.I.lets.get(xsem.wire_of(iw) or '', ''))):
            continue
        items = [xsem.wire_of(x) for x in xmodel.split_top(xmodel.strip_parens(lst)[1:-1], ', ')]
        if None in items:
            continue
        o = xsem.wire_of(ow)
        L = '[' + ', '.join(C.W(w) for w in items) + ']'
        name = f'sel_{a["field"]}'
        txt = f'/-- The selection `{a["field"]}` at the instance: member `i_0` of its list. -/\n'
        txt += HEAD.format(name=name, inst=C.inst)
        txt += f' :\n    {C.W(o)} = {L}.getD i_0 0 :=\n  select_getD (i := i_0) 0 {C.I.F(a)}\n'
        out.append((txt, name))
        for kn, (fin, S) in C.keyloops.items():
            if all(w in fin for w in items):
                L2 = '[' + ', '.join(f'({S}){tuple_proj(fin.index(w))}' for w in items) + ']'
                sk = f'selkeys_{a["field"]}'
                txt = f'/-- `{a["field"]}` through the branch-key sum of `{kn}`. -/\n'
                txt += HEAD.format(name=sk, inst=C.inst)
                txt += f' :\n    {C.W(o)} = {L2}.getD i_0 0 := by\n  rw [{name} F, ← {kn} F]\n'
                out.append((txt, sk))
                break
    return out


def tuple_proj(c):
    """Projection of component `c` of a right-nested tuple `(x0, x1, …, ())`."""
    return '.2' * c + '.1'


PRELUDE = '''import {module}_Facts
import Gen_{module}_Scopes
import Gen_InstCerts
import GenEnc
import GenBridge_Prf
import GenBridge_LutKey

/-! Generated by `generator/gen_instance_pre.py` (WP F): semantic facts of `{module}`'s instance
body `{inst}` at a symbolic instance `i_0` — PRF key cells (keys, tops, right-hand sides), LUT key
tables and members, root key loops, the decoder preimage and hash-sample links. Each statement is a
bridge lemma's conclusion instantiated at the instance (see the generator's docstring). -/

set_option maxHeartbeats 4000000
set_option maxRecDepth 16384
set_option linter.unusedVariables false

namespace RefreshTopProof.GenInst.{ns}Keys

open {module} RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLut
open RefreshTopProof.GenIface.BrLutKey RefreshTopProof.Gen
open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

/-! ## Generic lemmas (gadgets, column placements, slices, loop closed forms) -/

/-- The steps of an iteration satisfy `P` when every body step does. -/
theorem steps_of {{σ : Type}} {{body : Nat → σ → σ → Prop}} {{count : Nat}} {{init final : σ}}
    (h : MxxIR.IterRuns body count init final) {{P : Nat → σ → σ → Prop}}
    (hs : ∀ i cur next, body i cur next → P i cur next) : MxxIR.IterRuns P count init final :=
  RefreshTopProof.GenIface.seq_forall P hs h

open Classical in
/-- The runtime gadget matrix of `backend` (any width), or `0` if the backend has none. The one
term the preprocessing, online and final stages share for "the gadget". -/
noncomputable def gadgetOf (backend : BackendContext) (base digits : Int) (q n m : Nat) :
    ExactMatrix q n 1 m :=
  if h : ∃ G : ExactMatrix q n 1 m, gadgetMatrixRuns backend base digits G then h.choose else 0

/-- The gadget of modulus `q'` rebased to `q`. -/
noncomputable def rebasedGadget (backend : BackendContext) (base digits : Int) (q' q n m : Nat) :
    ExactMatrix q n 1 m :=
  centeredRebase (gadgetOf backend base digits q' n m)

theorem gadget_exists {{q n m : Nat}} {{backend : BackendContext}} {{base digits : Int}}
    (h : ∃ L : RegularLayout q, backend.regularLayout q n = some L ∧ base = L.base ∧
      digits = L.digitCount ∧ m = 1 * L.digitCount) :
    ∃ G : ExactMatrix q n 1 m, gadgetMatrixRuns backend base digits G := by
  obtain ⟨L, hL, hb, hd, hm⟩ := h
  exact ⟨_, L, hL, hb, hd, hm, rfl⟩

theorem gadgetOf_runs {{q n m : Nat}} {{backend : BackendContext}} {{base digits : Int}}
    (h : ∃ G : ExactMatrix q n 1 m, gadgetMatrixRuns backend base digits G) :
    gadgetMatrixRuns backend base digits (gadgetOf backend base digits q n m) := by
  unfold gadgetOf; rw [dif_pos h]; exact h.choose_spec

/-- Any runtime gadget matrix is `gadgetOf`. -/
theorem gadgetOf_eq {{q n m : Nat}} {{backend : BackendContext}} {{base digits : Int}}
    {{G : ExactMatrix q n 1 m}} (h : gadgetMatrixRuns backend base digits G) :
    gadgetOf backend base digits q n m = G :=
  gadgetMatrixRuns_unique (gadgetOf_runs ⟨G, h⟩) h

theorem rebasedGadget_runs {{q' q n m : Nat}} {{backend : BackendContext}} {{base digits : Int}}
    (hq : 1 < q) : centeredRebaseRuns (gadgetOf backend base digits q' n m)
      (rebasedGadget backend base digits q' q n m) := ⟨hq, rfl⟩

/-- Any rebase of a runtime gadget matrix is `rebasedGadget`. -/
theorem rebasedGadget_eq {{q' q n m : Nat}} {{backend : BackendContext}} {{base digits : Int}}
    {{G' : ExactMatrix q' n 1 m}} {{G : ExactMatrix q n 1 m}} (h' : gadgetMatrixRuns backend base digits G')
    (h : centeredRebaseRuns G' G) : rebasedGadget backend base digits q' q n m = G := by
  rw [rebasedGadget, gadgetOf_eq h', h.2]

section
variable {{n : Nat}}

/-- A root private key cell pair: `s • t_j + c_j = (pc μ (s, 1)_j) G + e_j`. -/
theorem root_cell {{q m : Nat}} {{sp : ErrorPoly n}} {{s o : ExactMatrix q n 1 1}}
    (hs : s = fun _ _ ↦ reducePoly q n sp) (ho : o = 1) (G top0 top1 : ExactMatrix q n 1 m) (pc : Int)
    {{pay : ExactMatrix q n 1 1}} {{μ : ErrorPoly n}} (hpay : pay = fun _ _ ↦ reducePoly q n μ)
    {{sigma : Rat}} {{cutoff : Int}} {{e0 e1 : ExactMatrix q n 1 m}}
    (he0 : gaussianSample sigma cutoff e0) (he1 : gaussianSample sigma cutoff e1) :
    ∃ e : Fin 2 → ErrorMatrix n 1 m, (∀ j, CoeffBound (e j) cutoff.toNat) ∧ ∀ j,
      reducePoly q n sp • ![top0, top1] j +
        ![matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight (matrixMulScalarLeft s G)
            (matrixPolynomial [pc])) pay) (matrixMulScalarLeft s top0)) e0,
          matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight (matrixMulScalarLeft o G)
            (matrixPolynomial [pc])) pay) (matrixMulScalarLeft s top1)) e1] j =
      reducePoly q n ((pc : ErrorPoly n) * μ * unitPair sp j) • G + reduceMatrix q n 1 m (e j) := by
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound he0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound he1
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  subst hs ho hpay hE0 hE1
  fin_cases j <;>
    simp only [matrixMulScalarLeft_eq, matrixMulScalarRight_eq, RefreshTopProof.Gen.matrixPolynomial_single_gen] <;>
    simp [unitPair, smul_smul, mul_comm, mul_left_comm, matrixAdd, matrixSub] <;> abel

/-- A runtime selection at a natural index is that member of the list. -/
theorem select_getD {{α : Type}} {{l : List α}} {{i : Nat}} {{out : α}} (d : α)
    (h : select (Int.ofNat i) l out) : out = l.getD i d := by
  obtain ⟨p, hp, rfl⟩ := h
  have hpi : (p : Nat) = i := by simp only [Int.ofNat_eq_natCast] at hp; exact_mod_cast hp
  subst hpi
  simp [List.getD_eq_getElem?_getD]

open Classical in
/-- A block `x` placed at block column `c` of a `C`-block row (the tensor with `unitRow c`). -/
noncomputable def colPlace (q n C m c : Nat) (x : ExactMatrix q n 1 m) : ExactMatrix q n (1 * 1) (C * m) :=
  if h : ∃ t, tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x t then h.choose else 0

theorem colPlace_eq {{q C m c : Nat}} {{x : ExactMatrix q n 1 m}} {{t : ExactMatrix q n (1 * 1) (C * m)}}
    (h : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x t) : t = colPlace q n C m c x := by
  have he : ∃ t, tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x t := ⟨t, h⟩
  unfold colPlace
  rw [dif_pos he]
  funext r p
  obtain rfl : r = 0 := Subsingleton.elim (α := Fin 1) _ _
  rw [tensorRuns_unitRow h p, tensorRuns_unitRow he.choose_spec p]

open Classical in
/-- The slice `[c0, c1)` of a row. -/
noncomputable def sliceOf (q n W m : Nat) (x : ExactMatrix q n 1 W) (c0 c1 : Int) : ExactMatrix q n 1 m :=
  if h : ∃ b : ExactMatrix q n 1 m, sliceMatrix x 0 1 c0 c1 b then h.choose else 0

theorem sliceOf_eq {{q W m : Nat}} {{x : ExactMatrix q n 1 W}} {{c0 c1 : Int}} {{b : ExactMatrix q n 1 m}}
    (h : sliceMatrix x 0 1 c0 c1 b) : b = sliceOf q n W m x c0 c1 := by
  have he : ∃ b : ExactMatrix q n 1 m, sliceMatrix x 0 1 c0 c1 b := ⟨b, h⟩
  unfold sliceOf
  rw [dif_pos he]
  exact sliceMatrix_unique h he.choose_spec

/-- A `Step::Sum` column loop: component `c` gains, in iteration `i`, the sum of the `W` members of
column `1 + i` placed at block `1 + i`. -/
theorem colsum_loop {{q C m F k W : Nat}} {{fams : Tup (Fin F → ExactMatrix q n 1 m) k}}
    {{body : Nat → Tup (ExactMatrix q n (1 * 1) (C * m)) k → Tup (ExactMatrix q n (1 * 1) (C * m)) k → Prop}}
    {{count : Nat}} {{init final : Tup (ExactMatrix q n (1 * 1) (C * m)) k}}
    (h : MxxIR.IterRuns body count init final)
    (hs : ∀ i cur next, body i cur next → ColStep.Spec fams W i cur next) (c : Fin k) :
    final.get c = init.get c + ∑ i ∈ Finset.range count,
      colPlace q n C m (1 + i) (∑ j ∈ Finset.range W, famAt (fams.get c) (W * (1 + i) + j)) := by
  refine MxxIR.IterRuns.invariant (Invariant := fun j s ↦ s.get c = init.get c + ∑ i ∈ Finset.range j,
      colPlace q n C m (1 + i) (∑ j ∈ Finset.range W, famAt (fams.get c) (W * (1 + i) + j))) (by simp) ?_ h
  intro i cur next hcur hb
  obtain ⟨-, -, hc⟩ := hs i cur next hb
  obtain ⟨t, ht, hn⟩ := hc c
  have ht' : tensorRuns (unitRow (((1 + i : Nat)) : Int) : ExactMatrix q n 1 C)
      (∑ j ∈ Finset.range W, famAt (fams.get c) (W * (1 + i) + j)) t := by
    have : ((1 + i : Nat) : Int) = 1 + (i : Int) := by push_cast; ring
    rw [this]; exact ht
  rw [hn, hcur, colPlace_eq ht', Finset.sum_range_succ, add_assoc]

/-- The slice form of a column loop: iteration `i` adds block `1 + i` of `x` times `v`, placed at `1 + i`. -/
theorem slicesum_loop {{q C m : Nat}} {{x : ExactMatrix q n 1 (C * m)}} {{v : ExactMatrix q n m 1}}
    {{body : Nat → ExactMatrix q n (1 * 1) (C * 1) → ExactMatrix q n (1 * 1) (C * 1) → Prop}}
    {{count : Nat}} {{init final : ExactMatrix q n (1 * 1) (C * 1)}}
    (h : MxxIR.IterRuns body count init final)
    (hs : ∀ i cur next, body i cur next → ColSliceStep.Spec x v i cur next) :
    final = init + ∑ i ∈ Finset.range count, colPlace q n C 1 (1 + i)
      (matrixMul (sliceOf q n (C * m) m x ((m : Int) * (1 + (i : Int))) ((m : Int) * (1 + (i : Int)) + m)) v) := by
  refine MxxIR.IterRuns.invariant (Invariant := fun j s ↦ s = init + ∑ i ∈ Finset.range j, colPlace q n C 1 (1 + i)
      (matrixMul (sliceOf q n (C * m) m x ((m : Int) * (1 + (i : Int))) ((m : Int) * (1 + (i : Int)) + m)) v))
    (by simp) ?_ h
  intro i cur next hcur hb
  obtain ⟨-, blk, t, hblk, ht, hn⟩ := hs i cur next hb
  have ht' : tensorRuns (unitRow (((1 + i : Nat)) : Int) : ExactMatrix q n 1 C) (matrixMul blk v) t := by
    have : ((1 + i : Nat) : Int) = 1 + (i : Int) := by push_cast; ring
    rw [this]; exact ht
  rw [hn, hcur, colPlace_eq ht', sliceOf_eq hblk, Finset.sum_range_succ, add_assoc]

/-- A member sum started from members `0` and `1` (iteration `i` adds member `2 + i`) is the sum of
the first `count + 2` members. -/
theorem memsum_full {{α : Type}} [AddCommMonoid α] {{F count : Nat}} {{fam : Fin F → α}} {{g0 g1 x0 fin : α}}
    (hg0 : familyGetStatic fam 0 g0) (hg1 : familyGetStatic fam 1 g1) (hx0 : x0 = g0 + g1)
    (hfin : fin = x0 + ∑ i ∈ Finset.range count, famAt fam (2 + Int.ofNat i).toNat) :
    fin = ∑ i ∈ Finset.range (count + 2), famAt fam i := by
  obtain ⟨p0, hp0, e0⟩ := hg0
  obtain ⟨p1, hp1, e1⟩ := hg1
  have h0 : g0 = famAt fam 0 := by
    rw [e0, famAt_of_lt _ (by omega)]; exact congrArg fam (Fin.ext (by simp at hp0; omega))
  have h1 : g1 = famAt fam 1 := by
    rw [e1, famAt_of_lt _ (by omega)]; exact congrArg fam (Fin.ext (by simp at hp1; omega))
  rw [hfin, hx0, h0, h1, Finset.sum_range_succ', Finset.sum_range_succ']
  have : ∑ i ∈ Finset.range count, famAt fam (2 + Int.ofNat i).toNat =
      ∑ i ∈ Finset.range count, famAt fam (i + 1 + 1) :=
    Finset.sum_congr rfl fun i _ ↦ congrArg _ (by simp only [Int.ofNat_eq_natCast]; omega)
  rw [this]; abel

/-- A column loop whose initial block is column `0` (member sum of the first `W` members, placed at
block `0`): all `count + 1` columns. -/
theorem colsum_full {{q C m F W count : Nat}} {{fam : Fin F → ExactMatrix q n 1 m}}
    {{col0 : ExactMatrix q n 1 m}} {{t0 fin : ExactMatrix q n (1 * 1) (C * m)}}
    (hcol0 : col0 = ∑ j ∈ Finset.range W, famAt fam j)
    (ht0 : tensorRuns (unitRow 0 : ExactMatrix q n 1 C) col0 t0)
    (hfin : fin = t0 + ∑ i ∈ Finset.range count,
      colPlace q n C m (1 + i) (∑ j ∈ Finset.range W, famAt fam (W * (1 + i) + j))) :
    fin = ∑ i ∈ Finset.range (count + 1),
      colPlace q n C m i (∑ j ∈ Finset.range W, famAt fam (W * i + j)) := by
  rw [hfin, colPlace_eq (c := 0) ht0, hcol0, Finset.sum_range_succ' _ count]
  simp only [Nat.mul_zero, Nat.zero_add, add_comm 1]
  rw [add_comm]

/-- The slice loop with its initial block `0`: all `count + 1` blocks. -/
theorem slicesum_full {{q C m count : Nat}} {{x : ExactMatrix q n 1 (C * m)}} {{v : ExactMatrix q n m 1}}
    {{b0 : ExactMatrix q n 1 m}} {{t0 fin : ExactMatrix q n (1 * 1) (C * 1)}}
    (hb0 : sliceMatrix x 0 1 0 m b0)
    (ht0 : tensorRuns (unitRow 0 : ExactMatrix q n 1 C) (matrixMul b0 v) t0)
    (hfin : fin = t0 + ∑ i ∈ Finset.range count, colPlace q n C 1 (1 + i)
      (matrixMul (sliceOf q n (C * m) m x ((m : Int) * (1 + (i : Int))) ((m : Int) * (1 + (i : Int)) + m)) v)) :
    fin = ∑ i ∈ Finset.range (count + 1), colPlace q n C 1 i
      (matrixMul (sliceOf q n (C * m) m x ((m : Int) * (i : Int)) ((m : Int) * (i : Int) + m)) v) := by
  rw [hfin, colPlace_eq (c := 0) ht0, sliceOf_eq hb0, Finset.sum_range_succ' _ count]
  have h0 : sliceOf q n (C * m) m x 0 (m : Int) =
      sliceOf q n (C * m) m x ((m : Int) * ((0 : Nat) : Int)) ((m : Int) * ((0 : Nat) : Int) + m) := by simp
  rw [h0, add_comm]
  congr 1
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [add_comm i 1]; push_cast; ring_nf

end

/-- The instances of the stage root. -/
abbrev S : Nat := {S}

'''


def gen_module(gdir, pdir, module, roles, only=None):
    C = Ctx(gdir, pdir, module, roles)
    ns = module.replace('Stage_', '').title().replace('_', '')
    parts, notes, names = [], [], []
    for a in C.ij['atoms']:
        L = a.get('loop')
        if not L or (only and a['field'] not in only):
            continue
        role = roles.get(L['child'], {}).get('role')
        try:
            if L['kind'] == 'par' and role in CELL_ROLES:
                txt, k = emit_par(C, a, 'keys')
                rtxt, rname = rhs_of(C, k)
                parts += [txt, rtxt]
                names += [k['name'], rname]
            elif L['kind'] == 'par' and role in TABLE_ROLES:
                txt, k = emit_par(C, a, 'table')
                parts.append(txt)
                names.append(k['name'])
            elif L['kind'] == 'par' and role in MEMBER_ROLES:
                txt, k = emit_par(C, a, 'member')
                parts.append(txt)
                names.append(k['name'])
            elif L['kind'] == 'seq' and role in KEYLOOP_ROLES:
                txt, n = emit_keyloop(C, a)
                parts.append(txt)
                names.append(n)
        except (KeyError, ValueError, IndexError, StopIteration) as e:
            notes.append(f'-- {a["field"]} ({role}) not generated: {e!r}')
    if not only:
        for a in C.ij['atoms']:
            L = a.get('loop')
            if L and L['kind'] == 'par' and L['child'] in C.sad:
                try:
                    txt, n = emit_struct_par(C, a)
                    parts.append(txt)
                    names.append(n)
                except (KeyError, ValueError, IndexError) as e:
                    notes.append(f'-- {a["field"]} (structural) not generated: {e!r}')
        seqs = [a for a in C.ij['atoms'] if a.get('loop') and a['loop']['kind'] == 'seq' and a['loop']['child'] in C.sad]
        memsums = {}
        for kind in ('MemberStep.Spec', 'ColStep.Spec', 'ColSliceStep.Spec'):
            for a in seqs:
                if not C.sad[a['loop']['child']][0].startswith(kind + ' '):
                    continue
                try:
                    if kind == 'MemberStep.Spec':
                        txt, n, d = emit_memsum(C, a)
                        memsums[n] = d
                    elif kind == 'ColStep.Spec':
                        txt, n = emit_colsum(C, a, memsums)
                    else:
                        txt, n = emit_slicesum(C, a)
                    parts.append(txt)
                    names.append(n)
                except (KeyError, ValueError, IndexError, AttributeError) as e:
                    notes.append(f'-- {a["field"]} ({kind}) not generated: {e!r}')
        try:
            extra = emit_bms(C) + emit_selects(C) + emit_target(C)
        except (KeyError, ValueError, IndexError) as e:
            extra = []
            notes.append(f'-- preimage target not generated: {e!r}')
        for txt, n in extra + emit_preimages(C) + emit_links(C) + emit_rootcells(C):
            parts.append(txt)
            names.append(n)
    out = PRELUDE.format(module=module, inst=C.inst, ns=ns, S=C.S) + C.cert_text() + '\n' + '\n'.join(parts)
    if notes:
        out += '\n' + '\n'.join(notes) + '\n'
    out += f'\nend RefreshTopProof.GenInst.{ns}Keys\n'
    return out, names, notes


if __name__ == '__main__':
    gdir, odir, pdir, rolesj = sys.argv[1:5]
    rest = sys.argv[5:]
    only = out_name = None
    if '--only' in rest:
        k = rest.index('--only')
        only = set(rest[k + 1].split(','))
        rest = rest[:k] + rest[k + 2:]
    if '--out' in rest:
        k = rest.index('--out')
        out_name = rest[k + 1]
        rest = rest[:k] + rest[k + 2:]
    mods = rest or ['Stage_preprocess']
    R = json.load(open(rolesj))
    for m in mods:
        out, names, notes = gen_module(gdir, pdir, m, R.get(m, {}), only)
        path = out_name or f'{odir}/Gen_Inst_{m}_Keys.lean'
        open(path, 'w').write(out)
        print(f'{m}: {len(names)} facts -> {path}, {len(notes)} skipped')
        for n in notes:
            print(' ', n)
