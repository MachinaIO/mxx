#!/usr/bin/env python3
"""Read-only structural metadata for Diamond iO 2 proof generators.

``extract(package)`` returns JSON-compatible metadata. ``python dio2_meta.py
PACKAGE... --output-dir DIR --check`` writes one JSON file per package and checks
the committed toy specialization when n=8. No Lean is generated or evaluated.

Wire identities are local to (view, module, scope). Ranges are half-open; large
families remain symbolic. Numeric constants are Python/JSON integers (consumers
must use arbitrary precision). Target ordering is the dio2 StepPlan contract,
checked against exported family sizes and the refresh/key dataflow. ``bindings``
retains the actual output wiring, including aliases in ExpLayout families.

The refresh_top parser supplies logical modules and expanded local notation.
Balanced output tuples are flattened here, as required by newer exports. Both
``generated`` and an optional ``worst`` view are inspected; modulus names are
never compared across modules, only their resolved values. Source digests detect
exports being replaced during extraction. Unsupported shapes fail explicitly.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'exponent-lut/lean/refresh_top/generator'))
import xmodel  # noqa: E402
from xmodules import module_sources  # noqa: E402


class MetadataError(ValueError):
    """The package is incomplete, inconsistent, or has an unsupported shape."""


def require(condition, message):
    if not condition:
        raise MetadataError(message)


def one(values, description):
    values = list(values)
    require(len(values) == 1, f'{description}: expected one, got {len(values)}')
    return values[0]


def integer(text):
    return int(xmodel.strip_parens(text))


def dims(ty):
    m = re.search(r'ExactMatrix (\w+) (\d+) (\d+) (\d+)', ty or '')
    return (m[1], *map(int, m.groups()[1:])) if m else None


def family_count(ty):
    m = re.search(r'Fin (\d+) →', ty or '')
    return int(m[1]) if m else 1


def output_wires(scope):
    # Output equations contain a pure tuple of wire names, possibly balanced.
    term = one((a.args[0] for a in scope.atoms if a.head == 'out'),
               f'{scope.module}.{scope.name} output')
    require(not re.sub(r'w_\d+(?:_\d+)?|[\s(),]', '', term),
            f'Unsupported output expression in {scope.name}')
    return xmodel.WIRE.findall(term)


def flat_tuple(term):
    """Flatten tuple branches, preserving applications and non-tuple leaves."""
    term = xmodel.strip_parens(term)
    if not term:
        return []
    parts = xmodel.split_top(term, ',')
    if len(parts) == 1:
        return [term]
    return [leaf for part in parts for leaf in flat_tuple(part)]


class Module:
    def __init__(self, directory, name, fingerprints):
        self.name = name
        sources = module_sources(directory, name)
        self.moduli = {}
        for physical, source in sources:
            path = directory / (physical + '.lean')
            fingerprints[path] = hashlib.sha256(source.encode()).hexdigest()
            expanded = xmodel.expand_matrix_notations(source)
            for alias, value in re.findall(r'^abbrev (modulus_\d+) : Nat := (\d+)$', expanded, re.M):
                value = int(value)
                require(alias not in self.moduli or self.moduli[alias] == value,
                        f'{name}: conflicting modulus {alias}')
                self.moduli[alias] = value
        self.scopes = xmodel.parse_module(directory, name)
        self.root = self.scopes['generatedRoot']
        self.parents = {}
        for scope in self.scopes.values():
            for atom in scope.loops:
                loop = atom.loop
                if loop.child in self.scopes:
                    # xmodel's old tuple parser preserves left tuple branches.
                    loop.args = [v for a in loop.args for v in flat_tuple(a)]
                    loop.outs = [v for a in loop.outs for v in flat_tuple(a)]
                    self.parents[loop.child] = (scope, loop)

    def dimension(self, scope, wire):
        result = dims(scope.wires[wire].type)
        require(result is not None, f'{self.name}.{scope.name}.{wire}: no matrix type')
        modulus, n, rows, columns = result
        return dict(modulus=self.moduli.get(modulus, modulus), n=n, rows=rows, columns=columns)

    def bindings(self, names):
        wires = output_wires(self.root)
        require(len(wires) == len(names), f'{self.name}: stage output arity mismatch {len(wires)} != {len(names)}')
        return {name: dict(wire=w, members=family_count(self.root.wires[w].type),
                           matrix=self.dimension(self.root, w) if dims(self.root.wires[w].type) else None)
                for name, w in zip(names, wires)}


class Flow:
    """Backward producer lookup; never relies on wire numbers or offsets."""
    def __init__(self, scope):
        self.scope = scope
        self.producers = {}
        for a in scope.atoms:
            if a.head.startswith('MxxRuntime.') and a.args and xmodel.WIRE.fullmatch(a.args[-1]):
                self.producers[a.args[-1]] = a

    def op(self, wire, head):
        w = self.scope.wires[wire]
        actual, args = xmodel.app_split(w.term or '')
        require(actual == 'MxxRuntime.' + head,
                f'{self.scope.name}.{wire}: expected {head}, got {actual}')
        return args

    def atom(self, wire, head):
        a = self.producers.get(wire)
        require(a is not None and a.kind == head,
                f'{self.scope.name}.{wire}: expected predicate {head}')
        return a.args[:-1]

    def constant(self, wire):
        term = self.scope.wires[wire].term or ''
        m = re.search(r'matrixPolynomial \[\((-?\d+)\)\]', term)
        require(m is not None, f'{wire}: expected scalar polynomial constant')
        return int(m[1])

    def leaves(self, wire):
        head, args = xmodel.app_split(self.scope.wires[wire].term or '')
        if head == 'MxxRuntime.matrixAdd':
            return self.leaves(args[0]) + self.leaves(args[1])
        return [wire]

    def ancestry(self, wire):
        seen, pending = set(), [wire]
        while pending:
            w = pending.pop()
            if w in seen:
                continue
            seen.add(w)
            a = self.producers.get(w)
            if a:
                refs = xmodel.WIRE.findall(' '.join(a.args[:-1]))
            else:
                refs = xmodel.WIRE.findall(self.scope.wires[w].term or '')
            pending.extend(v for v in refs if v != w and v in self.scope.wires)
        return seen


def refresh_metadata(module):
    candidates = [s for s in module.scopes.values()
                  if s.atom_kinds().get('modulusSwitchRuns', 0) > 1]
    s = one(candidates, module.name + ' refresh part')
    f = Flow(s)
    result = one(output_wires(s), 'refresh result')
    parent, loop = module.parents[s.name]
    require(parent is module.root, 'refresh parts must be a root family')
    primes = []
    for term in f.leaves(result):
        lift, basis_wire = f.op(term, 'matrixMulScalarRight')
        switch = f.atom(lift, 'centeredRebaseRuns')[0]
        zeta = f.atom(switch, 'modulusSwitchRuns')[0]
        z1, half = f.op(zeta, 'matrixSub')
        reduced, total = f.op(z1, 'matrixSub')
        bp = f.atom(reduced, 'modulusReduceRuns')[0]
        anchor, preimage = f.op(bp, 'matrixMul')
        cf, smudge = f.op(total, 'matrixAdd')
        combination, fresh = f.op(cf, 'matrixAdd')
        gcd, cxy = f.op(combination, 'matrixSub')
        gc, digits = f.op(gcd, 'matrixMul')
        cx, y = f.op(cxy, 'matrixMul')
        g, c1 = f.atom(gc, 'concatColumns')
        digit_raw = f.atom(digits, 'compactCenteredRebaseRuns')[1]
        backend, base, count, za = f.atom(digit_raw, 'gadgetDecomposeRuns')
        quotient_wire, target = f.op(za, 'matrixMulScalarLeft')
        yr = f.atom(y, 'compactCenteredRebaseRuns')[1]
        _, ybase, ycount, zy = f.atom(yr, 'gadgetDecomposeRuns')
        ylift, cp = f.op(zy, 'matrixMulScalarRight')
        ysmall = f.atom(ylift, 'centeredRebaseRuns')[0]
        yy, s10 = f.op(ysmall, 'matrixMulScalarRight')
        yg, s9 = f.op(yy, 'matrixMulScalarRight')
        gadget_lift = f.atom(yg, 'modulusReduceRuns')[0]
        gadget = f.atom(gadget_lift, 'centeredRebaseRuns')[0]
        f.atom(gadget, 'gadgetMatrixRuns')
        q = module.dimension(s, switch)['modulus']
        q0 = module.dimension(s, term)['modulus']
        quotient, basis = f.constant(quotient_wire), f.constant(basis_wire)
        require(q0 == q * quotient, f'{s.name}.{term}: Q0 != q * quotient')
        require(basis % q == 1 and basis % quotient == 0,
                f'{s.name}.{term}: invalid CRT basis')
        require((base, count) == (ybase, ycount), 'inconsistent decoder digit shapes')
        named = dict(term=term, lift=lift, switch=switch, zeta=zeta, half=half,
                     reduced=reduced, anchor=anchor, preimage=preimage, fresh=fresh,
                     smudge=smudge, combination=combination, g=g, c1=c1, cx=cx,
                     digits=digits, raw_digits=digit_raw, target=target, y=y,
                     raw_y=yr, gadget=gadget, gadget_lift=gadget_lift,
                     quotient=quotient_wire, basis=basis_wire)
        primes.append(dict(index=len(primes), prime=q, quotient=quotient, crt_basis=basis,
                           cprime=f.constant(cp), s9=f.constant(s9), s10=f.constant(s10),
                           base=integer(base), digits=integer(count), wires=named,
                           dimensions={k: module.dimension(s, w) for k, w in named.items()},
                           constraints={k: dict(fragment=f.producers[w].frag, field=f.producers[w].field)
                                  for k, w in named.items() if w in f.producers}))
    qs = [p['prime'] for p in primes]
    require(len(qs) == len(set(qs)), 'duplicate CRT primes')
    require(math.prod(qs) == q0, 'CRT product does not equal refresh modulus')
    require(sum(p['crt_basis'] for p in primes) % q0 == 1, 'CRT basis sum is not one')
    return dict(module=module.name, scope=s.name, parts=loop.count,
                targets=loop.count // 2, components=2,
                part_index='2 * target + component', result_wire=result,
                family_output=loop.outs, arguments=loop.args, primes=primes,
                Q0=q0, ring_dimension=module.dimension(s, result)['n'])


def prf_metadata(module, n):
    """Identify root row-sum families, grouping blocks by member count."""
    groups = defaultdict(list)
    for a in module.root.loops:
        loop = a.loop
        s = module.scopes.get(loop.child)
        if s is None or loop.kind != 'par':
            continue
        kids = [a.loop for a in s.loops]
        if len(kids) == 2 and [k.kind for k in kids] == ['par', 'seq'] and \
                kids[1].count == kids[0].count - 2 and \
                s.atom_kinds().get('familyGetStatic', 0) > 0:
            groups[loop.count].append(dict(scope=s.name, bucket_width=kids[0].count,
                                           arguments=loop.args, outputs=loop.outs))
    require(len(groups) == 3, f'{module.name}: expected sigma, CBD, smudge PRF groups, got {list(groups)}')
    counts = sorted(groups)
    require(counts[0] == n, 'sigma member count does not equal ring dimension')
    return {name: dict(members=count, packed_members=count // n,
                       blocks=len(groups[count]), bucket_widths=[g['bucket_width'] for g in groups[count]],
                       block_scopes=groups[count])
            for name, count in zip(['sigma', 'cbd', 'smudge'], counts)}


def source_tables(module):
    tables = {}
    for s in module.scopes.values():
        for w in s.wires.values():
            if not w.term or 'selectInt' not in w.term:
                continue
            m = re.search(r'selectInt .*?\[([^\]]*)\]', w.term)
            if m and re.fullmatch(r'[\s(),\d-]*', m[1]):
                values = list(map(int, re.findall(r'-?\d+', m[1])))
                key = json.dumps(values)
                tables.setdefault(key, dict(values=values, uses=[]))['uses'].append(
                    dict(scope=s.name, wire=w.name))
    return list(tables.values())


def keys_metadata(module, names, refresh):
    bindings = module.bindings(names)
    by_out = {w: (s, l) for s, l in module.parents.values()
              for o in l.outs for w in xmodel.WIRE.findall(o)[:1]}
    results = []
    for p in refresh['primes']:
        label = f'dio2/key/{p["index"]}'
        b = bindings[label]
        require(b['members'] == refresh['parts'], f'{label}: key family count mismatch')
        require(b['wire'] in by_out, f'{label}: missing key producer')
        parent, loop = by_out[b['wire']]
        require(parent is module.root, f'{label}: not a root family')
        child = module.scopes[loop.child]
        require(b['matrix']['modulus'] == refresh['Q0'], f'{label}: key modulus mismatch')
        flow = Flow(child)
        quotient_uses = []
        for w in child.wires.values():
            head, args = xmodel.app_split(w.term or '')
            if head == 'MxxRuntime.matrixMulScalarLeft':
                constant_term = child.wires[args[0]].term or ''
                if 'matrixPolynomial' in constant_term and flow.constant(args[0]) == p['quotient']:
                    quotient_uses.append(dict(product=w.name, scalar=args[0], target=args[1]))
        require(quotient_uses, f'{label}: no dataflow using its CRT quotient')
        prime_moduli = {module.moduli[d[0]] for w in child.wires.values() if (d := dims(w.type))}
        require(p['prime'] in prime_moduli, f'{label}: prime modulus absent in key body')
        results.append(dict(prime_index=p['index'], label=label, wire=b['wire'],
                            scope=child.name, members=loop.count, arguments=loop.args,
                            quotient_uses=quotient_uses, output_wires=output_wires(child)))
    return results


def scalar(scope, term, member):
    """Evaluate only exported integer routing expressions, never matrix operations."""
    t = xmodel.strip_parens(term)
    if re.fullmatch(r'-?\d+', t):
        return int(t)
    if t in ('i_0', 'Int.ofNat i_0'):
        return member
    if t in scope.wires:
        w = scope.wires[t]
        if w.term:
            return scalar(scope, w.term, member)
        a = Flow(scope).producers.get(t)
        require(a is not None and a.kind == 'select', f'Unsupported integer producer {t}')
        items = xmodel.split_top(a.args[1].strip('[]'), ',')
        return scalar(scope, items[scalar(scope, a.args[0], member)], member)
    if t.startswith('decide '):
        return scalar(scope, t[7:], member)
    if t.startswith('if '):
        cond, rest = t[3:].split(' then ', 1)
        yes, no = rest.split(' else ', 1)
        return scalar(scope, yes if scalar(scope, cond, member) else no, member)
    for op in [' + ', ' - ', ' * ', ' / ', ' % ', ' < ', ' = ']:
        parts = xmodel.split_top(t, op)
        if len(parts) == 2:
            a, b = (scalar(scope, p, member) for p in parts)
            if op == ' + ': return a + b
            if op == ' - ': return a - b
            if op == ' * ': return a * b
            if op == ' / ': return a // b
            if op == ' % ': return a % b
            if op == ' < ': return a < b
            if op == ' = ': return a == b
    head, args = xmodel.app_split(t)
    if head == 'MxxIR.selectInt':
        entries = xmodel.split_top(args[1].strip('[]'), ',')
        return scalar(scope, entries[scalar(scope, args[0], member)], member)
    raise MetadataError(f'Unsupported integer routing expression {t}')


def root_routes(module, bindings, refresh, labels, auts):
    """Discover root-to-part routes and check semantic class boundary members.

    Compact routes retain the exact exported equations. Boundary checks are not
    claimed as universal proofs; generators can use those equations for Lean.
    """
    root, flow = module.root, Flow(module.root)
    family = xmodel.WIRE.findall(refresh['family_output'][0])[0]
    loops = {xmodel.WIRE.findall(o)[0]: loop for parent, loop in module.parents.values() if parent is root
             for o in loop.outs if xmodel.WIRE.findall(o)}

    def route(wire):
        a = flow.producers.get(wire)
        if a and a.kind == 'familyGetStatic' and a.args[0] == family:
            return dict(indices=[integer(a.args[1])])
        if a and a.kind == 'familyPack':
            children = [route(w) for w in xmodel.WIRE.findall(a.args[1])]
            return dict(indices=[one(c['indices'], 'packed root member') for c in children])
        loop = loops.get(wire)
        require(loop is not None, f'Cannot discover root route for {wire}')
        child = module.scopes[loop.child]
        child_flow = Flow(child)
        out = one(output_wires(child), 'root router output')
        read = child_flow.producers.get(out)
        require(read is not None and read.kind in ('familyGetDynamic', 'familyGetStatic'),
                f'{child.name}: output is not a family read')
        require(loop.args == [family], f'{child.name}: router does not consume refresh family')
        return dict(scope=child.name, members=loop.count, index=read.args[1],
                    expressions={w.name:w.term for w in child.wires.values() if w.type in ('Int','Bool')},
                    selections=[a.to_json() for a in child.atoms if a.kind == 'select'])

    helper_start = 3 + labels['rows'] * labels['target_columns']
    result = {}
    for name, binding in bindings.items():
        if not name.startswith('rns-bgg/root-'):
            continue
        r = route(binding['wire'])
        count = binding['members']
        # Check each finite Aut/ExpLayout table, but only label boundary members.
        members = range(count) if '/root-helper/' in name else sorted({0, count - 1, min(count - 1, labels['columns'])})
        port = int(name.rsplit('.', 1)[1])
        samples = []
        for member in members:
            index = r['indices'][member] if 'indices' in r else scalar(module.scopes[r['scope']], r['index'], member)
            require(0 <= index < refresh['parts'] and index % 2 == port, f'{name}: invalid part routing')
            expected = None
            if '/Aut(' in name:
                aut = int(re.search(r'Aut\((\d+)\)', name)[1])
                expected = helper_start + 1 + auts.index(aut)
            elif '/one-private.' in name:
                expected = 0
            elif '/state.' in name:
                expected = 2
            elif '/label-0.' in name:
                expected = 1 if member == labels['rows'] else 3 + member * labels['target_columns']
            elif '/label.' in name:
                expected = 4 + (member // labels['columns']) * labels['target_columns'] + member % labels['columns']
            if expected is not None:
                require(index == 2 * expected + port, f'{name}[{member}]: target order mismatch')
            samples.append(dict(member=member, part=index, target=index // 2))
        r['checked_members'] = samples
        result[name] = r
    return result


def backend_metadata(package, fingerprints, refresh):
    path = package / 'generated/Backend.lean'
    source = path.read_text()
    fingerprints[path] = hashlib.sha256(source.encode()).hexdigest()
    lists = {name: list(map(int, re.findall(r'\d+', values)))
             for name,values in re.findall(r'def (moduli\d+) : List Nat := \[([^]]*)\]', source)}
    layouts = []
    for m in re.finditer(r'def (layout\d+) : MxxRuntime.RegularLayout (\d+) :=(.*?)(?=\ndef |\Z)', source, re.S):
        fields = dict(re.findall(r'(crtModuli|droppedModuli|baseBits|base|digitsPerTower) := (\w+)', m[3]))
        layout = {k: (lists[v] if k == 'crtModuli' else int(v)) for k,v in fields.items()}
        layout.update(name=m[1], modulus=int(m[2]))
        require(math.prod(layout['crtModuli'][layout['droppedModuli']:]) == layout['modulus'], 'backend layout product mismatch')
        layouts.append(layout)
    primary = one((l for l in layouts if l['modulus'] == refresh['Q0']), 'backend Q0 layout')
    require(primary['crtModuli'] == [p['prime'] for p in refresh['primes']], 'backend/refresh CRT order mismatch')
    require(primary['base'] == refresh['primes'][0]['base'], 'backend gadget base mismatch')
    require(primary['digitsPerTower'] * len(primary['crtModuli']) == refresh['primes'][0]['digits'], 'backend gadget digits mismatch')
    return layouts


def structural_scopes(module):
    """Keep symbolic label forests and family routing as compact scope equations."""
    result = []
    for s in module.scopes.values():
        hash_labels = any('108, 97, 98, 101, 108' in a.text for a in s.atoms if a.kind == 'hashSample')
        if s.name != 'generatedRoot' and (hash_labels or any(
                a.kind == 'familyGetDynamic' for a in s.atoms)):
            parent = module.parents.get(s.name)
            result.append(dict(scope=s.name, count=parent[1].count if parent else None,
                               parent=parent[0].name if parent else None,
                               label_hashes=hash_labels,
                               loops=[a.loop.to_json() for a in s.loops if a.loop.child in module.scopes],
                               routing=[a.to_json() for a in s.atoms if a.kind in ('familyGetDynamic', 'select')],
                               expressions={w.name: w.term for w in s.wires.values()
                                            if w.src == 'let' and w.type in ('Int', 'Bool')}))
    return result


def extract(package):
    """Extract and validate one package without changing it or enumerating families."""
    package = Path(package).resolve()
    if package.name == 'generated':
        package = package.parent
    fingerprints = {}
    stages_path = package / 'generated/stages.json'
    stages_bytes = stages_path.read_bytes()
    fingerprints[stages_path] = hashlib.sha256(stages_bytes).hexdigest()
    stages = json.loads(stages_bytes)
    views = {}
    for view, prefix in [('generated', ''), ('worst', 'W_')]:
        directory = package / view
        if not directory.is_dir():
            continue
        modules = {}
        for stage in stages:
            name = prefix + 'Stage_' + stage
            require((directory / (name + '.lean')).exists(), f'Missing {view}/{name}.lean')
            modules[stage] = Module(directory, name, fingerprints)
        requirement = Module(directory, prefix + 'Requirement_0', fingerprints)
        claim_path = directory / (prefix + 'Claim.lean')
        claim = claim_path.read_bytes()
        fingerprints[claim_path] = hashlib.sha256(claim).hexdigest()
        claim_calls = re.findall(r'\b(' + prefix + r'Stage_\w+)\.generatedRoot\b', claim.decode())
        require(set(claim_calls) == {m.name for m in modules.values()}, 'Claim stage set mismatch')
        refresh = refresh_metadata(modules['online_0'])
        n = refresh['ring_dimension']
        bindings = {stage: m.bindings(stages[stage]['outputs']) for stage, m in modules.items()}
        # Every matrix in every scope, including Requirement, must use the same ring.
        dimensions = {d[1] for m in [*modules.values(), requirement] for s in m.scopes.values()
                      for w in s.wires.values() if (d := dims(w.type))}
        require(dimensions == {n}, f'{view}: inconsistent ring dimensions {dimensions}')
        initial = bindings['initial']
        label0 = initial['rns-bgg/root-row/dio2/label-0.0']['members']
        label_members = initial['rns-bgg/root-row/dio2/label.0']['members']
        rows = label0 - 1  # The final label-0 member is the public root.
        require(rows > 0 and label_members % rows == 0, 'invalid label rectangle')
        width = label_members // rows
        auts = sorted({int(a) for name in initial for a in re.findall(r'/Aut\((\d+)\)\.', name)})
        label_targets = rows * (width + 1)
        helper_start = 3 + label_targets
        anchor_start = helper_start + 1 + len(auts)  # sigma itself is Aut(1).
        require(refresh['parts'] % 2 == 0 and refresh['targets'] > anchor_start,
                'target family does not cover labels, helpers and anchors')
        ranges = [dict(kind=k, start=i, stop=i + 1) for i, k in enumerate(['private', 'public', 'state'])]
        ranges += [dict(kind='labels', start=3, stop=helper_start, rows=rows, columns=width + 1,
                        index='3 + row * columns + column'),
                   dict(kind='sigma', automorphism=1, start=helper_start, stop=helper_start + 1)]
        ranges += [dict(kind='Aut', automorphism=a, start=helper_start + i + 1,
                        stop=helper_start + i + 2) for i, a in enumerate(auts)]
        ranges += [dict(kind='anchor', start=anchor_start, stop=refresh['targets'])]
        prfs = prf_metadata(modules['online_0'], n)
        key_prfs = prf_metadata(modules['keys_online_0_0'], n)
        require({k: (v['members'], v['bucket_widths']) for k,v in prfs.items()} ==
                {k: (v['members'], v['bucket_widths']) for k,v in key_prfs.items()},
                'online/key PRF shapes disagree')
        columns = refresh['primes'][0]['digits']
        require(prfs['cbd']['packed_members'] == refresh['parts'] * columns, 'CBD layout mismatch')
        divisor = prfs['cbd']['members'] * len(refresh['primes'])
        smudge_digits, rem = divmod(prfs['smudge']['members'], divisor)
        require(rem == 0 and smudge_digits > 0, 'smudge layout mismatch')
        gadgets = defaultdict(list)
        for stage, m in modules.items():
            for s in m.scopes.values():
                for a in s.atoms:
                    if a.kind in ('gadgetMatrixRuns', 'gadgetDecomposeRuns'):
                        base, digits = map(integer, a.args[1:3])
                        d = m.dimension(s, a.args[-1])
                        gadgets[(d['modulus'], base, digits)].append(dict(stage=stage, scope=s.name,
                                                                         wire=a.args[-1], operation=a.kind))
        level_artifacts = defaultdict(list)
        for stage, entries in bindings.items():
            for name, entry in entries.items():
                match = re.match(r'lut-helper/(\d+)/', name)
                if match:
                    level_artifacts[int(match[1])].append(dict(stage=stage, name=name, **entry))
        levels = [dict(level=level, columns=sorted({a['matrix']['columns'] for a in artifacts}),
                       gadgets=[dict(base=b, digits=d) for q,b,d in gadgets
                                if q in {a['matrix']['modulus'] for a in artifacts}],
                       artifacts=artifacts) for level,artifacts in sorted(level_artifacts.items())]
        # Dio2Parameters uses a constant-modulus chain and L = levels - 2.
        # The largest exported helper level is L; L+1 is its state gadget.
        # Keep this contract-derived entry distinct from artifact-bearing levels.
        require(levels and [x['level'] for x in levels] == list(range(len(levels))),
                'non-contiguous exported helper levels')
        require(len(gadgets) == 1, 'unsupported non-constant dio2 gadget chain')
        terminal = dict(level=len(levels), columns=[columns],
                        gadgets=[dict(base=refresh['primes'][0]['base'], digits=columns)],
                        artifacts=[], provenance='Dio2Parameters: state gadget at L+1, levels=L+2')
        levels.append(terminal)
        views[view] = dict(refresh=refresh, ring_dimension=n, Q0=refresh['Q0'],
                           primes=[p['prime'] for p in refresh['primes']],
                           columns=columns, anchor_columns=refresh['primes'][0]['dimensions']['anchor']['columns'],
                           smudge_digits=smudge_digits, automorphisms=auts,
                           labels=dict(rows=rows, columns=width, target_columns=width + 1,
                                       members=label_members, root_members=label0),
                           target_classes=ranges, prfs=prfs, key_prfs=key_prfs,
                           levels=levels, level_count=len(levels), artifact_level_count=len(levels)-1,
                           level_count_provenance='max exported helper level + 2 (Dio2Parameters contract)',
                           keys=keys_metadata(modules['keys_online_0_0'], stages['keys_online_0_0']['outputs'], refresh),
                           gadgets=[dict(modulus=q, base=b, digits=d, uses=uses)
                                    for (q,b,d),uses in gadgets.items()], bindings=bindings,
                           family_scopes={stage: structural_scopes(modules[stage])
                                                for stage in ['initial', 'online_0', 'keys_online_0_0']},
                           lut_tables={stage: source_tables(m) for stage,m in modules.items()
                                       if stage.startswith(('preprocess', 'online', 'keys_online'))},
                           requirement=dict(module=requirement.name, scopes=list(requirement.scopes)),
                           claim_stages=claim_calls)
        forests = {}
        for stage in ['initial', 'online_0', 'keys_online_0_0']:
            m = modules[stage]
            selected = {entry['scope'] for entry in views[view]['family_scopes'][stage] if entry['label_hashes']}
            for name in list(selected):
                while name in m.parents:
                    name = m.parents[name][0].name
                    if name != 'generatedRoot':
                        selected.add(name)
            pending = list(selected)
            while pending:
                name = pending.pop()
                for child in m.scopes[name].children:
                    if child in m.scopes and child not in selected:
                        selected.add(child)
                        pending.append(child)
            forests[stage] = [dict(scope=name, count=m.parents[name][1].count if name in m.parents else None,
                                   children=m.scopes[name].children) for name in sorted(selected)]
            # Legacy toy labels are unrolled in the root, so record their local hash wires.
            if not forests[stage]:
                hashes = [a.args[-1] for a in m.root.atoms if a.kind == 'hashSample'
                          and '108, 97, 98, 101, 108' in a.text]
                forests[stage] = [dict(scope='generatedRoot', label_hash_wires=hashes, unrolled=True)]
        views[view]['label_forest_scopes'] = forests
        views[view]['root_routes'] = root_routes(modules['online_0'], bindings['online_0'], refresh,
                                                 views[view]['labels'], auts)
        # Stage outputs used as preimage inputs have the same family/matrix shape.
        for p in refresh['primes']:
            b = bindings['preimages_0_0'][f'dio2/preimage/{p["index"]}']
            require(b['members'] == refresh['parts'], 'preimage family count mismatch')
            require(b['matrix'] == p['dimensions']['preimage'], 'preimage/online type mismatch')
        # Root objects survive refresh with the same member and matrix shapes.
        for name, b in initial.items():
            if name.startswith('rns-bgg/root-'):
                other = bindings['online_0'][name]
                require((b['members'], b['matrix']) == (other['members'], other['matrix']),
                        f'initial/online root shape mismatch: {name}')
    require('generated' in views, 'Missing generated view')
    if 'worst' in views:
        for key in ['ring_dimension', 'Q0', 'primes', 'columns', 'anchor_columns',
                    'smudge_digits', 'automorphisms', 'labels', 'target_classes', 'level_count']:
            require(views['generated'][key] == views['worst'][key], f'generated/worst mismatch: {key}')
    backend = backend_metadata(package, fingerprints, views['generated']['refresh'])
    for path, digest in fingerprints.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f'Export changed during extraction; retry: {path}')
    return dict(schema_version=1, package=package.name, backend_layouts=backend, views=views,
                sources={str(p.relative_to(package)): h for p,h in sorted(fingerprints.items())})


def check(metadata):
    """Regression certificate for the published toy proof, not a Lean proof."""
    v = metadata['views']['generated']
    if v['ring_dimension'] == 8:
        expected = dict(primes=5, parts=36, targets=18, columns=25, levels=6, auts=7,
                        cbd=7200, smudge=36000, cbd_packed=900, smudge_packed=4500)
        actual = dict(primes=len(v['primes']), parts=v['refresh']['parts'], targets=v['refresh']['targets'],
                      columns=v['columns'], levels=v['level_count'], auts=len(v['automorphisms']),
                      cbd=v['prfs']['cbd']['members'], smudge=v['prfs']['smudge']['members'],
                      cbd_packed=v['prfs']['cbd']['packed_members'],
                      smudge_packed=v['prfs']['smudge']['packed_members'])
        require(actual == expected, f'toy specialization mismatch: {actual}')
        require(v['automorphisms'] == [3,5,7,9,11,13,15], 'toy automorphisms mismatch')
        require(all(p['blocks'] == 2 and set(p['bucket_widths']) == {2} for p in v['prfs'].values()),
                'toy PRF shape mismatch')
    return dict(package=metadata['package'], ring_dimension=v['ring_dimension'],
                primes=len(v['primes']), Q0_bits=v['Q0'].bit_length(),
                columns=v['columns'], levels=v['level_count'], parts=v['refresh']['parts'], targets=v['refresh']['targets'],
                labels=v['labels'], automorphisms=len(v['automorphisms']), smudge_digits=v['smudge_digits'],
                prfs={k: dict(members=p['members'], packed_members=p['packed_members'],
                              blocks=p['blocks'], bucket_widths=sorted(set(p['bucket_widths'])))
                      for k,p in v['prfs'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packages', type=Path, nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for package in args.packages:
        metadata = extract(package)
        summary = check(metadata) if args.check else dict(package=metadata['package'])
        destination = args.output_dir / (metadata['package'] + '.json')
        destination.write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
