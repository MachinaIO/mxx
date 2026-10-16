#!/usr/bin/env python3
"""Regenerate step-zero refresh, symbolic-label, and helper proof modules.

The old scalar proofs are templates, not sources of package parameters. Wire and
Facts projections are matched through the exported producer graph, stopping at
upstream relation hypotheses rather than expanding their sampler/LUT producers.
Forest loops remain symbolic. Use --output-dir for a production dry run without
touching the production package.
"""

import argparse
import json
import re
from pathlib import Path

import dio2_meta as meta


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'average-n8-p5x50'
WIRE = meta.xmodel.WIRE


def clean(source):
    source = re.sub(r'/\-.*?\-/', '', source, flags=re.S)
    return '\n'.join(line for line in source.splitlines() if not line.lstrip().startswith('--'))


def facts(directory, scope):
    source = '\n'.join(path.read_text() for path in sorted(directory.glob('W_Stage_online_0_Facts*.lean')))
    result = {}
    pattern = rf'^structure {scope}\.Facts_(\d+)\b[^\n]*\n(.*?)(?=^structure |^theorem |^end|\Z)'
    for match in re.finditer(pattern, source, re.M | re.S):
        fields = re.findall(r'^  (\w+)\s*:', match[2], re.M)
        for index, field in enumerate(fields):
            result[f'c{match[1]}_{index}'] = f'c{match[1]}.{field}'
    return result


class Port:
    def __init__(self, old, new, oldfacts, newfacts, params):
        self.old, self.new = old, new
        self.oldfacts, self.newfacts = oldfacts, newfacts
        self.params = params
        self.mapping = {}
        self.atoms = {}
        self.producers = []
        for scope in (old, new):
            outputs = {}
            for atom in scope.atoms:
                if atom.loop:
                    names = WIRE.findall(atom.loop.final if atom.loop.kind == 'seq' else ' '.join(atom.loop.outs))
                elif atom.head.startswith('MxxRuntime.') and atom.args:
                    names = WIRE.findall(atom.args[-1])
                else:
                    names = []
                for name in names:
                    outputs[name] = atom
            self.producers.append(outputs)

    def wire(self, left, right):
        if left in self.mapping:
            meta.require(self.mapping[left] == right, f'conflicting producer match {left}: {self.mapping[left]} != {right}')
            return
        self.mapping[left] = right
        source, target = self.old.wires[left], self.new.wires[right]
        if source.term and target.term and not source.term.startswith(('inputs.', 'witness.')):
            lefthead, leftargs = meta.xmodel.app_split(source.term)
            righthead, rightargs = meta.xmodel.app_split(target.term)
            meta.require(lefthead == righthead, f'producer changed for {left}/{right}: {lefthead}/{righthead}')
            self.refs(source.term, target.term)
        elif left in self.producers[0] and right in self.producers[1]:
            self.atom(self.producers[0][left], self.producers[1][right])

    def refs(self, left, right):
        oldrefs, newrefs = WIRE.findall(left), WIRE.findall(right)
        meta.require(len(oldrefs) == len(newrefs), f'producer arity changed: {left[:100]} / {right[:100]}')
        for source, target in zip(oldrefs, newrefs):
            if source in self.old.wires and target in self.new.wires:
                self.wire(source, target)

    def atom(self, left, right):
        key = left.field
        if key in self.atoms:
            meta.require(self.atoms[key] == right.field, f'conflicting atom {key}')
            return
        if left.kind != right.kind:
            return
        self.atoms[key] = right.field
        if left.loop:
            if left.loop.count == 0:
                meta.require(right.loop.count == 0, 'one-branch helper no longer has an empty loop')
            self.refs(' '.join(left.loop.outs), ' '.join(right.loop.outs))
            if left.loop.kind == 'seq':
                self.refs(left.loop.final, right.loop.final)
            if left.loop.count == 0:
                self.refs(' '.join(left.loop.args), ' '.join(right.loop.args))
                self.refs(left.loop.init, right.loop.init)
        else:
            self.refs(' '.join(left.args), ' '.join(right.args))

    def render(self, source):
        reverse = {field: name for name, field in self.oldfacts.items()}
        byfield = {atom.field: atom for atom in self.old.atoms}
        requested = re.findall(r'F\.(c\d+\.\w+)', source)
        for name in requested:
            key = reverse[name]
            if key not in self.atoms:
                atom = byfield[key]
                candidates = []
                for target in self.new.atoms:
                    if atom.kind != target.kind:
                        continue
                    oldrefs, newrefs = WIRE.findall(atom.text), WIRE.findall(target.text)
                    if len(oldrefs) != len(newrefs):
                        continue
                    pairs = [(self.mapping[old], new) for old, new in zip(oldrefs, newrefs) if old in self.mapping]
                    if pairs and all(old == new for old, new in pairs):
                        candidates.append(target)
                self.atom(atom, meta.one(candidates, f'constraint {name}'))
        source = re.sub(r'F\.(c\d+\.\w+)', lambda match: 'F.' + self.newfacts[self.atoms[reverse[match[1]]]], source)
        source = re.sub(r'generatedRoot\.(w_\d+(?:_\d+)?(?:_concat_\d+)?)',
                        lambda match: 'generatedRoot.' + self.mapping[match[1]], source)
        for old, new in self.params.items():
            source = re.sub(rf'(?<![\w_]){old}(?![\w_])', str(new), source)
        return source


def section(source, start, stop):
    return source[source.index(start):source.index(stop)]


LABEL_GENERIC = r'''
namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenIface RefreshTopProof.GenFinal

namespace Online0Labels

variable {n q m : Nat}

/-- One symbolic forest node: both ciphertext rows, four key blocks, and its
gain-one public plaintext. The plaintext is tied to the exported payload. -/
def Node (sm : ErrorPoly n) (G : ExactMatrix q n 1 m) (B : Nat)
    (c0 c1 k00 k01 k10 k11 : ExactMatrix q n 1 m) (payload : ExactMatrix q n 1 1) : Prop :=
  ∃ α : ErrorPoly n, payload = (fun _ _ ↦ reducePoly q n α) ∧ SmulGain α 1 ∧
    Rel sm ![![k00, k01], ![k10, k11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![c0, c1] B

/-- Carrying an unmatched node never resets or drops its encryption error. -/
theorem Node.mono {sm : ErrorPoly n} {G : ExactMatrix q n 1 m} {B C : Nat}
    {c0 c1 k00 k01 k10 k11 : ExactMatrix q n 1 m} {payload : ExactMatrix q n 1 1}
    (h : Node sm G B c0 c1 k00 k01 k10 k11 payload) (hBC : B ≤ C) :
    Node sm G C c0 c1 k00 k01 k10 k11 payload := by
  obtain ⟨α, hp, hα, hr⟩ := h
  exact ⟨α, hp, hα, hr.mono hBC⟩

/-- Dynamic reads at the same exported integer select the same member. -/
theorem read_at {α : Type} {W : Nat} {family : Fin W → α} {index : Int} {value : α}
    (member : Fin W) (hindex : (member.val : Int) = index)
    (h : familyGetDynamic family index value) : value = family member := by
  obtain ⟨position, hposition, hvalue⟩ := h
  have heq : position = member := Fin.ext (by omega)
  simpa only [heq] using hvalue

/-- Selection keeps either the carried node or the merged node, without a
case split on any member of the label family. -/
theorem select_pair {α : Type} {flag : Bool} {left right value : α}
    (h : select (if flag then 1 else 0) [left, right] value) :
    value = if flag then right else left := by
  obtain ⟨position, hposition, hvalue⟩ := h
  cases flag
  · have heq : position = (0 : Fin 2) := Fin.ext (by simpa using hposition)
    simpa only [heq, List.get_eq_getElem] using hvalue
  · have heq : position = (1 : Fin 2) := Fin.ext (by simpa using hposition)
    simpa only [heq, List.get_eq_getElem] using hvalue

/-- Binary forest multiplication. Only the inherited left error is acted on
by dense digits; the right error has gain one, not the ring dimension. -/
theorem merge (hn : 0 < n) (hq : 1 < q) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm : ErrorPoly n} {B : Nat}
    {a0 a1 a00 a01 a10 a11 b0 b1 b00 b01 b10 b11 : ExactMatrix q n 1 m}
    {ap bp : ExactMatrix q n 1 1}
    (ha : Node sm G B a0 a1 a00 a01 a10 a11 ap)
    (hb : Node sm G B b0 b1 b00 b01 b10 b11 bp)
    {d00 d01 d10 d11 : ExactMatrix q n m m}
    (hd00 : gadgetDecomposeRuns backend base digits b00 d00)
    (hd01 : gadgetDecomposeRuns backend base digits b01 d01)
    (hd10 : gadgetDecomposeRuns backend base digits b10 d10)
    (hd11 : gadgetDecomposeRuns backend base digits b11 d11) :
    Node sm G (m * n * B * (base.toNat / 2) + m * n * B * (base.toNat / 2) + B)
      (a0 * d00 + a1 * d10 + matrixMulScalarLeft ap b0)
      (a0 * d01 + a1 * d11 + matrixMulScalarLeft ap b1)
      (a00 * d00 + a01 * d10) (a00 * d01 + a01 * d11)
      (a10 * d00 + a11 * d10) (a10 * d01 + a11 * d11)
      (matrixMulScalarLeft ap bp) := by
  obtain ⟨α, hap, hα, har⟩ := ha
  obtain ⟨β, hbp, hβ, hbr⟩ := hb
  have hd0 : ∀ j, gadgetDecomposeRuns backend base digits (![b00, b01] j) (![d00, d01] j) :=
    by intro j; fin_cases j <;> assumption
  have hd1 : ∀ j, gadgetDecomposeRuns backend base digits (![b10, b11] j) (![d10, d11] j) :=
    by intro j; fin_cases j <;> assumption
  have hr := mul_public_split_mono hn hq layout hlayout hexact hG hα har hbr hd0 hd1
  refine ⟨α * β, payload_mul hap hbp, smulGain_mul hα hβ, ?_⟩
  have ht : (fun j ↦ reducePoly q n α • (reducePoly q n (β * unitPair sm j) • G)) =
      (fun j ↦ reducePoly q n ((α * β) * unitPair sm j) • G) := by
    funext j
    rw [smul_smul, ← map_mul, mul_assoc]
  rw [ht] at hr
  have hc : ∀ row : ExactMatrix q n 1 m,
      matrixMulScalarLeft ap row = reducePoly q n α • row := by
    intro row
    rw [hap]
    exact matrixMulScalarLeft_eq _ _
  simp only [hc]
  convert hr using 1 <;> first | rfl |
    (funext first second; fin_cases first <;> fin_cases second <;> rfl)

end Online0Labels

'''


def label_branch(scope):
    import gen_adapters as adapters
    binders, call, names = adapters.header(scope)
    term = lambda wire: adapters.wire_term(scope, wire)
    inputs = [wire for wire in scope.wires.values() if wire.src == 'input']
    families = [wire for wire in inputs if wire.type.startswith('Fin ')]
    meta.require(len(families) == 7, 'binary forest must carry seven families')
    size = meta.family_count(families[0].type)
    bywire = {atom.args[-1]: atom for atom in scope.atoms if atom.kind == 'familyGetDynamic'}
    selections = [atom for atom in scope.atoms if atom.kind == 'select']
    meta.require(len(selections) == 7, 'binary forest must select seven outputs')
    inputwire = {wire.name: wire.term for wire in inputs}
    reads = [atom for atom in scope.atoms if atom.kind == 'familyGetDynamic']
    indices = list(dict.fromkeys(atom.args[1] for atom in reads))
    meta.require(len(indices) == 2, 'binary forest must read left and right members')
    def node_args(index):
        found = {atom.args[0]: atom.args[-1] for atom in reads if atom.args[1] == index}
        meta.require(len(found) == 7, 'binary forest member has incomplete row/key/payload reads')
        return found
    left, right = [node_args(index) for index in indices]
    row_family = [bywire[meta.xmodel.split_top(meta.xmodel.strip_parens(selections[j].args[1])[1:-1], ',')[0]].args[0]
                  for j in range(6)]
    payload_family = bywire[meta.xmodel.split_top(meta.xmodel.strip_parens(selections[6].args[1])[1:-1], ',')[0]].args[0]
    ordered = row_family + [payload_family]
    inv = ' '.join(f'({inputwire[family]} member)' for family in ordered)
    outs = ' '.join(f'(outputs{'.2' * index}.1)' for index in range(7))
    per, flat = adapters.atom_names(scope)
    per = adapters.use_rfl_out(per, flat)
    factmap = {atom.field: name for atom, (name, _) in zip(scope.atoms, flat)}
    proof = [f'  obtain {adapters.pattern(scope, per)} := h']
    for side, index in zip(['left', 'right'], indices):
        first = next(atom for atom in reads if atom.args[1] == index)
        proof.append(f'  obtain ⟨{side}Member, {side}Index, {side}Read⟩ := {factmap[first.field]}')
        for atom in reads:
            if atom.args[1] == index:
                fact = factmap[atom.field]
                if atom is first:
                    proof.append(f'  have get_{atom.args[-1]} := {side}Read')
                else:
                    proof.append(f'  have get_{atom.args[-1]} := Online0Labels.read_at {side}Member {side}Index {fact}')
    for side, table in zip(['left', 'right'], [left, right]):
        proof.append(f'  have h{side} := hinput {side}Member')
        proof.append(f'  rw [{", ".join("← get_" + table[family] for family in ordered)}] at h{side}')
    decompose = {atom.args[-2]: atom for atom in scope.atoms if atom.kind == 'gadgetDecomposeRuns'}
    decomposition = [decompose[right[family]] for family in [ordered[2], ordered[3], ordered[4], ordered[5]]]
    proof.append('  have hmerge := Online0Labels.merge hn hq layout hlayout hexact hG hleft hright ' +
                 ' '.join(factmap[atom.field] for atom in decomposition))
    proof.append('  have hcarry := hleft.mono (Nat.le_add_left B (m * n * B * (base.toNat / 2) + m * n * B * (base.toNat / 2)))')
    for atom in selections:
        proof.append(f'  have select_{atom.args[-1]} := Online0Labels.select_pair {factmap[atom.field]}')
    proof.append('  rw [' + ', '.join('select_' + atom.args[-1] for atom in selections) + ']')
    flag = scope.wires[selections[0].args[0]].term
    condition = scope.wires[WIRE.findall(flag)[0]].term.removeprefix('decide ')
    condition = term(condition)
    proof.append(f'  by_cases hactive : {condition}')
    proof.append('  · simpa only [hactive, decide_true, ↓reduceIte] using hmerge')
    proof.append('  · simpa only [hactive, decide_false, Bool.false_eq_true, ↓reduceIte] using hcarry')
    declaration = f'''/-- One exported binary-forest iteration, for an arbitrary member. -/
theorem {scope.name}_node {binders}
    {{sm : ErrorPoly n}} {{G : ExactMatrix modulus_0 n 1 m}} {{B : Nat}}
    (hn : 0 < n) (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 n = some layout) (hexact : layout.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend base digits G)
    (hinput : ∀ member : Fin {size}, Online0Labels.Node sm G B {inv})
    (h : {call}) :
    Online0Labels.Node sm G (m * n * B * (base.toNat / 2) + m * n * B * (base.toNat / 2) + B)
      {outs} := by
''' + '\n'.join(proof) + '\n'
    return declaration


def label_module(module, view, directory):
    import gen_adapters as adapters
    forest = meta.one((entry for entry in view['label_forest_scopes']['online_0'] if entry['children']), 'label forest')
    scope = module.scopes[forest['scope']]
    branches = [module.scopes[name] for name in forest['children'][1:]]
    n, m = view['ring_dimension'], view['columns']
    prime = view['refresh']['primes'][0]
    base, digits = prime['base'], prime['digits']
    localfacts = facts(directory, scope.name)
    generated = []
    for branch in branches:
        declaration = label_branch(branch)
        for symbol, value in [('n', str(n)), ('m', str(m)), ('base', f'({base} : Int)'), ('digits', str(digits))]:
            declaration = re.sub(rf'\b{symbol}\b', value, declaration)
        generated.append(declaration)
    flow = meta.Flow(scope)
    output = meta.output_wires(scope)
    meta.require(len(output) == 6, 'forest output must contain two rows and four keys')
    mul0, pub0 = flow.op(output[0], 'matrixAdd')
    left0, right0 = flow.op(mul0, 'matrixAdd')
    leftg, d00 = flow.op(left0, 'matrixMul')
    rightg, d10 = flow.op(right0, 'matrixMul')
    c0, dg = flow.op(leftg, 'matrixMul')
    c1, dg1 = flow.op(rightg, 'matrixMul')
    meta.require(dg == dg1, 'forest retarget must be shared')
    payload, cs0 = flow.op(pub0, 'matrixMulScalarLeft')
    mul1, pub1 = flow.op(output[1], 'matrixAdd')
    _, cs1 = flow.op(pub1, 'matrixMulScalarLeft')
    left1, right1 = flow.op(mul1, 'matrixAdd')
    _, d01 = flow.op(left1, 'matrixMul')
    _, d11 = flow.op(right1, 'matrixMul')
    matrix = lambda wire: f'{scope.name}.{wire} ctx'
    keywires = [flow.atom(wire, 'gadgetDecomposeRuns')[-1] for wire in [d00, d01, d10, d11]]
    rawdg = flow.atom(dg, 'compactCenteredRebaseRuns')[-1]
    form = flow.atom(rawdg, 'gadgetDecomposeRuns')[-1]
    rebased, _ = flow.op(form, 'matrixMulScalarRight')
    G = flow.atom(rebased, 'centeredRebaseRuns')[0]
    rootkeys = []
    for wire in output[2::2]:
        for summand in flow.op(wire, 'matrixAdd'):
            intermediate, _ = flow.op(summand, 'matrixMul')
            key, _ = flow.op(intermediate, 'matrixMul')
            rootkeys.append(key)
    nodewires = [c0, c1] + rootkeys + [payload]
    def family(loop, member):
        wires = WIRE.findall(' '.join(loop.outs))
        meta.require(len(wires) == 7, 'forest level must output seven families')
        return ' '.join(f'({matrix(wire)} {member})' for wire in wires)
    def fact(atom):
        return 'F.' + localfacts[atom.field]
    def producer(wire):
        return flow.producers[wire]
    loops = [next(atom for atom in scope.loops if atom.loop.child == child) for child in forest['children']]
    bound = 'B'
    assembly = []
    for level, atom in enumerate(loops[1:], 1):
        bound = f'({m} * {n} * {bound} * {base // 2} + {m} * {n} * {bound} * {base // 2} + {bound})'
        boundname = f'forestBound_{level}'
        assembly.append(f'  let {boundname} : Nat := {bound}')
        assembly.append(f'  have nodes_{level} : ∀ index : Fin {atom.loop.count},\n'
                        f'      Online0Labels.Node sm ({matrix(G)}) {boundname} {family(atom.loop, "index")} := by\n'
                        f'    intro index\n'
                        f'    exact {atom.loop.child}_node hn hq layout hlayout hexact hG '
                        f'{"hleaves" if level == 1 else "nodes_" + str(level - 1)} ({fact(atom)} index)')
        bound = boundname
    meta.require(loops[-1].loop.count == 1, 'forest must finish with one root')
    assembly.append(f'  have hroot := nodes_{len(loops) - 1} (0 : Fin 1)')
    for wire in nodewires:
        assembly.append(f'  have get_{wire} := Online0Labels.read_at (0 : Fin 1) rfl {fact(producer(wire))}')
    assembly.append('  rw [' + ', '.join('← get_' + wire for wire in nodewires) + '] at hroot')
    assembly.append('  obtain ⟨α, hpayload, hα, hrel⟩ := hroot')
    assembly.append(f'  have hform : {matrix(form)} = {matrix(G)} := gadget_rebase_one hq {fact(producer(rebased))}')
    assembly.append(f'  have hdg := {fact(producer(rawdg))}')
    assembly.append('  rw [hform] at hdg')
    assembly.append(f'  have hd : {matrix(dg)} = {matrix(rawdg)} := compactRebase_self_gen hn hq {fact(producer(dg))}')
    assembly.append('  rw [← hd] at hdg')
    for side, ds in [('d0', [d00, d01]), ('d1', [d10, d11])]:
        keys = keywires[:2] if side == 'd0' else keywires[2:]
        assembly.append(f'  have h{side} : ∀ j : Fin 2, gadgetDecomposeRuns backend {base} {digits}\n'
                        f'      (![{matrix(keys[0])}, {matrix(keys[1])}] j) (![{matrix(ds[0])}, {matrix(ds[1])}] j) := by\n'
                        f'    intro j; fin_cases j\n'
                        f'    · exact {fact(producer(ds[0]))}\n'
                        f'    · exact {fact(producer(ds[1]))}')
    assembly.append('  have hfinish := label_finish_mono hn hq layout hlayout hexact hG hα hrel hdg hs hd0 hd1')
    assembly.append('  refine ⟨α, hpayload, hα, ?_⟩')
    outfact = next(atom for atom in scope.atoms if atom.kind == 'out')
    assembly.append(f'  rw [{fact(outfact)}]')
    assembly.append('  have hscalar : ∀ row : ExactMatrix modulus_0 ' + str(n) + ' 1 ' + str(m) + ',\n'
                    f'      matrixMulScalarLeft ({matrix(payload)}) row = reducePoly modulus_0 {n} α • row := by\n'
                    '    intro row\n    rw [hpayload]\n    rfl')
    for component, ds, ciphertext in [(0, [d00, d10], cs0), (1, [d01, d11], cs1)]:
        expression = f'{matrix(c0)} * {matrix(dg)} * {matrix(ds[0])} + {matrix(c1)} * {matrix(dg)} * {matrix(ds[1])}'
        assembly.append(f'  have hc{component} : {matrix(output[component])} = {expression} + reducePoly modulus_0 {n} α • {matrix(ciphertext)} := by\n'
                        f'    change {expression} + matrixMulScalarLeft ({matrix(payload)}) ({matrix(ciphertext)}) = _\n'
                        '    rw [hscalar]')
    assembly.append('  convert hfinish using 1 <;> first | rfl | (funext first second; fin_cases first <;> fin_cases second <;> rfl) | (funext component; fin_cases component; exact hc0; exact hc1)')
    beta = base // 2
    declarations = '\n'.join(f'def online0_label_bound_{level} (B : Nat) : Nat := ' +
                            (f'B' if level == 0 else f'{m} * {n} * online0_label_bound_{level-1} B * {beta} + {m} * {n} * online0_label_bound_{level-1} B * {beta} + online0_label_bound_{level-1} B')
                            for level in range(len(loops)))
    final = f'({m} * {n} * ({m} * {n} * online0_label_bound_{len(loops)-1} B * {beta}) * {beta} + {m} * {n} * ({m} * {n} * online0_label_bound_{len(loops)-1} B * {beta}) * {beta} + Bs)'
    for level in range(1, len(loops)):
        old = next(line for line in assembly if line.startswith(f'  let forestBound_{level} '))
        assembly[assembly.index(old)] = f'  let forestBound_{level} : Nat := online0_label_bound_{level} B'
    statement = f'''open W_Stage_online_0 in
/-- All binary forest levels and the final refresh action for an arbitrary label
member. Initial/Import and Requirement must establish the selected leaf nodes;
this theorem proves every subsequent forest multiplication and retarget. -/
theorem online0_label_target {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (member : Fin {forest['count']}) (ctx : {scope.name}.Ctx backend hashModel params member.val)
    {{outputs : _}} (F : {scope.name}.Facts ctx outputs)
    {{sm : ErrorPoly {n}}} {{B Bs : Nat}} {{ts : Fin 2 → ExactMatrix modulus_0 {n} 1 {m}}}
    (hn : 0 < {n}) (hq : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 {n} = some layout) (hexact : layout.droppedModuli = 0)
    (hG : gadgetMatrixRuns backend {base} {digits} ({matrix(G)}))
    (hleaves : ∀ index : Fin {loops[0].loop.count}, Online0Labels.Node sm ({matrix(G)}) B {family(loops[0].loop, 'index')})
    (hs : Rel sm ![![{matrix(keywires[0])}, {matrix(keywires[1])}], ![{matrix(keywires[2])}, {matrix(keywires[3])}]]
      ts ![{matrix(cs0)}, {matrix(cs1)}] Bs) :
    ∃ α : ErrorPoly {n}, {matrix(payload)} = (fun _ _ ↦ reducePoly modulus_0 {n} α) ∧ SmulGain α 1 ∧
      Rel sm ![![outputs.2.2.1, outputs.2.2.2.1], ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]]
        (fun j ↦ reducePoly modulus_0 {n} α • ts j) ![outputs.1, outputs.2.1] {final} := by
''' + '\n'.join(assembly) + '\n'
    header = '''import Dio2Sharp
import Dio2Label
import Dio2Online0Refresh

/-!
# Symbolic step-zero label forests

Generated by `generator/gen_online0_targets.py`. A family member is arbitrary;
each logarithmic forest level is proved uniformly from its dynamic reads and
selection constraints. The only forest-input hypothesis is the selected leaf
invariant (Initial/Import plus Requirement's public-sign semantics). The final
plaintext equals the exported payload and has gain one. The sigma/public pair
and backend exactness are explicit upstream hypotheses.
-/

set_option maxHeartbeats 16000000
set_option maxRecDepth 16384

'''
    result = header + LABEL_GENERIC + 'open W_Stage_online_0\n\n' + '\n'.join(generated) + '\n' + declarations + '\n\n' + statement + '\nend Dio2Proof\n'
    return result.replace('modulus_0', meta.dims(scope.wires[c0].type)[0])


def helper_module(original, module, oldfacts, newfacts, view, params, targets):
    source = clean((OLD / 'proof/Dio2Online0Helper.lean').read_text())
    generic = section(source, 'namespace Dio2Proof', 'open W_Stage_online_0 in\nset_option maxHeartbeats 64000000')
    gadget = section(generic, 'open W_Stage_online_0 in', 'set_option maxHeartbeats 32000000 in')
    generic = generic.replace(gadget, '')
    for old, new in params.items():
        generic = re.sub(rf'(?<![\w_]){old}(?![\w_])', str(new), generic)
    n = view['ring_dimension']
    meta.require(n > 0 and n & (n - 1) == 0, 'LUT dimension must be a power of two')
    generic = generic.replace('(log := 3)', f'(log := {n.bit_length() - 1})')
    port = Port(original.root, module.root, oldfacts, newfacts, params)
    for old, new in zip(['w_237_0', 'w_238_0', 'w_244_0', 'w_245_0'], targets[:4]):
        port.wire(old, new)
    gadget = port.render(gadget)
    generic = generic.replace('set_option maxHeartbeats 32000000 in', gadget + 'set_option maxHeartbeats 32000000 in', 1)
    prefix = [atom.args[1] for atom in module.root.atoms if atom.kind == 'hashSample'
              and '108, 117, 116, 45, 97' in atom.text]
    prefix = meta.one(set(prefix), 'helper hash prefix')
    generic += f'\nnamespace Online0Helpers\n\nabbrev helperPrefix : Blob := {prefix}\n\nend Online0Helpers\n\n'
    template = section(source, 'theorem online0_aut_3 ', 'open W_Stage_online_0 in\nset_option maxHeartbeats 64000000 in\n\ntheorem online0_aut_5 ')
    declarations = []
    claims = []
    for index, automorphism in enumerate(view['automorphisms']):
        port = Port(original.root, module.root, oldfacts, newfacts, params)
        port.wire('w_811_0', targets[8 + 2 * index])
        port.wire('w_812_0', targets[9 + 2 * index])
        statement = port.render(template).replace('online0_aut_3', f'online0_aut_{automorphism}')
        statement = re.sub(r'\b3\b', str(automorphism), statement)
        statement = statement.replace('⟨1, rfl⟩', f'⟨{(automorphism - 1) // 2}, rfl⟩')
        statement = statement.replace('helperPre', 'Online0Helpers.helperPrefix')
        declarations.append('open W_Stage_online_0 in\nset_option maxHeartbeats 64000000 in\n' + statement.strip())
        signature = statement.split(':= by', 1)[0]
        binders, conclusion = signature.split(' :\n    Rel', 1)
        binders = binders.removeprefix(f'theorem online0_aut_{automorphism} ')
        claims.append(f'open W_Stage_online_0 in\nset_option linter.unusedVariables false in\ndef online0_aut_claim_{automorphism} : Prop :=\n  ∀ {binders.strip()},\n    Rel{conclusion.rstrip()}')
    names = [f'online0_aut_claim_{automorphism}' for automorphism in view['automorphisms']]
    meta.require(bool(names), 'the exported automorphism set is empty')
    certificates = f'@online0_aut_{view["automorphisms"][-1]}'
    for automorphism in reversed(view['automorphisms'][:-1]):
        certificates = f'⟨@online0_aut_{automorphism}, {certificates}⟩'
    indexed = '\n\n'.join(claims) + f'''

/-- Export order, taken from metadata rather than a presumed consecutive range. -/
def online0_automorphisms : List Nat := {view['automorphisms']}

theorem online0_automorphisms_length : online0_automorphisms.length = {len(names)} := rfl

/-- One actual exported automorphism, with the metadata-set order preserved. -/
def online0_automorphism (member : Fin {len(names)}) : Nat :=
  online0_automorphisms.get ⟨member.val, by simpa only [online0_automorphisms_length] using member.isLt⟩

/-- The exact root-context propositions for the individually exported circuits. -/
def online0_aut_claims : List Prop := [{', '.join(names)}]

/-- Lookup of a proved exported Aut target at an arbitrary metadata-set index.
The finite table follows the already distinct exported circuits; lookup does not
enumerate or case-split on the index, labels, coefficients, or ring members. -/
theorem online0_aut_target (member : Fin {len(names)}) :
    online0_aut_claims.get ⟨member.val, by simpa only [online0_aut_claims, List.length_cons,
      List.length_nil] using member.isLt⟩ := by
  have allClaims : List.Forall id online0_aut_claims := {certificates}
  exact (List.forall_iff_forall_mem.mp allClaims) _ (List.get_mem _ _)
'''
    sigma = source[source.index('theorem online0_target_sigma '):source.rindex('end Dio2Proof')]
    port = Port(original.root, module.root, oldfacts, newfacts, params)
    port.wire('w_753_0', targets[6])
    port.wire('w_754_0', targets[7])
    port.wire('w_237_0', targets[0])
    sigma = port.render(sigma)
    result = '''import Dio2LutBridge
import Dio2Online0Refresh

/-!
# Step-zero sigma and exported automorphism targets

Generated by `generator/gen_online0_targets.py`. Each exported Aut circuit is
proved from the generic one-branch LUT lemma. The sigma sampler relation, helper
root relation, switching-ciphertext relation, Walsh identity, plaintext bound,
and exact backend layout remain explicit upstream hypotheses. No label, part,
digit, coefficient, or ring-member family is enumerated.
-/

set_option maxRecDepth 16384

''' + generic.strip() + '\n\n' + '\n\n'.join(declarations) + '\n\n' + indexed + '\n\nopen W_Stage_online_0 in\nset_option maxHeartbeats 32000000 in\n' + sigma.strip() + '\n\nend Dio2Proof\n'
    return result.replace('modulus_0', meta.dims(module.root.wires[targets[0]].type)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text()) if args.metadata else meta.extract(args.package)
    view = metadata['views']['worst']
    n, columns = view['ring_dimension'], view['columns']
    base = view['refresh']['primes'][0]['base']
    params = {8: n, 25: columns, 4096: base, 2048: base // 2}
    module = meta.Module(args.package / 'worst', 'W_Stage_online_0', {})
    original = meta.Module(OLD / 'worst', 'W_Stage_online_0', {})
    oldfacts = facts(OLD / 'worst', 'generatedRoot')
    newfacts = facts(args.package / 'worst', 'generatedRoot')
    root = module.root
    packs = [atom for atom in root.atoms if atom.kind == 'familyPack']
    scalar_count = 2 * (view['refresh']['targets'] - view['labels']['rows'] * view['labels']['target_columns'])
    packs = [atom for atom in packs if meta.integer(atom.args[0]) == scalar_count]
    pack = min(packs, key=lambda atom: int(atom.args[-1].split('_')[1]))
    targets = WIRE.findall(pack.args[1])
    meta.require(len(targets) == scalar_count, 'scalar target pack changed')
    refresh_source = clean((OLD / 'proof/Dio2Online0Refresh.lean').read_text())
    refresh_source = refresh_source[refresh_source.index('namespace Dio2Proof'):]
    port = Port(original.root, root, oldfacts, newfacts, params)
    for old, new in zip(['w_237_0', 'w_238_0', 'w_244_0', 'w_245_0', 'w_356_0', 'w_371_0'], targets[:6]):
        port.wire(old, new)
    refresh = port.render(refresh_source)
    refresh = refresh.replace('modulus_0', meta.dims(root.wires[targets[0]].type)[0])
    member = view['labels']['rows']
    refresh = re.sub(r'(generatedRoot\.' + port.mapping['w_17_0'] + r' ctx) 1\b', rf'\1 {member}', refresh)
    header = '''import Dio2Rel
import Dio2Generic
import Gen_W_Stage_online_0_Scopes
import W_Stage_online_0_Facts

/-!
# Step-zero scalar refresh inputs and targets

Generated by `generator/gen_online0_targets.py` from package metadata and the
exported producer graph. Sigma, initial-root, and state-LUT encoding invariants
are explicit hypotheses, supplied by Dio2Online0Sigma, Dio2Initial/Import, and
Dio2Online0State. Exact backend layout remains an explicit caller hypothesis;
no new axiom or sampler specification is introduced here.
-/

set_option maxRecDepth 16384

'''
    output = args.output_dir or args.package / 'proof'
    helper = helper_module(original, module, oldfacts, newfacts, view, params, targets)
    label = label_module(module, view, args.package / 'worst')
    modules = {'Dio2Online0Refresh': header + refresh.strip() + '\n',
               'Dio2Online0Label': label, 'Dio2Online0Helper': helper}
    output.mkdir(parents=True, exist_ok=True)
    for name, text in modules.items():
        (output / (name + '.lean')).write_text(text)
    if args.output_dir:
        (output / 'online0-targets-wiring.json').write_text(json.dumps(dict(parameters=params,
            scalar_targets=targets, automorphisms=view['automorphisms'],
            refresh_wire_map=port.mapping), indent=2) + '\n')
    print(json.dumps(dict(package=metadata['package'], modules=['Dio2Online0Refresh', 'Dio2Online0Label', 'Dio2Online0Helper'], scalar_targets=len(targets))))


if __name__ == '__main__':
    main()
