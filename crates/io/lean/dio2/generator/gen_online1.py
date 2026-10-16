#!/usr/bin/env python3
"""Regenerate the step-1 PRF, packing, digit, state and key root adapters.

Common proofs quantify family sizes. Exported geometry, wires, scopes and tags
are obtained through dio2_meta and validated against its metadata. --out permits
production generation in scratch without touching or building production.
"""
import argparse
from pathlib import Path
import re
from dio2_meta import extract, Module, Flow, one, output_wires, require, family_count

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / 'average-n8-p5x50-v2/proof'
COMMON = ['Dio2Online1Prf', 'Dio2KeyAgree1', 'Dio2SmudgeValues', 'Dio2SmudgeHalf']
MARKER = '/- Generated final-output adapter: gen_online1.py. -/'
PRF_MARKER = '/- Generated smudge adapters: gen_online1.py. -/'


def tuple_of(values):
    result = '()'
    for value in reversed(values):
        result = f'({value}, {result})'
    return result


def root_fact(module, wire):
    candidates = [a for a in module.root.atoms if
                  (a.loop and wire in ' '.join(a.loop.outs + [a.loop.final or ''])) or
                  (not a.loop and a.args and a.args[-1] == wire)]
    atom = one(candidates, f'producer of {wire}')
    return f'F.c{atom.frag}.h{wire[2:]}'


def prf_buckets(package, metadata):
    """Instantiate each bucket from its dataflow, keeping all member families symbolic."""
    module = Module(package / 'worst', 'W_Stage_online_1', {})
    buckets = [a for a in module.root.loops if a.loop.kind == 'par' and
               len(a.loop.args) == 6 and
               'matrixAdd' in ' '.join(w.term or '' for w in
                                      module.scopes[a.loop.child].wires.values())]
    buckets = [a for a in buckets if
               any(b.loop and b.loop.kind == 'gather' and
                   b.loop.outs == [arg + ' i' for arg in a.loop.args]
                   for b in module.root.loops)]
    require(buckets, 'no step-1 PRF buckets')
    old = (HERE.parent / 'average-n8-p5x50/proof/Dio2Online1Prf.lean').read_text()
    templates = [old[old.index('theorem online1_bucket0'):old.index('/-- The tag prefix of the step-1 smudge PRF\'s second')],
                 old[old.index('theorem online1_bucket1'):old.index('/-- The tag prefix of the LUT helpers')]]
    view = metadata['views']['worst']
    layout = one((item for item in metadata['backend_layouts'] if
                  item['modulus'] == view['Q0']), 'step-1 layout')
    result = PRF_MARKER + '\nopen W_Stage_online_1\nset_option maxRecDepth 16384\n'
    for index, bucket in enumerate(buckets):
        summed = bucket.loop
        cell = one((a for a in module.root.loops if a.loop.kind == 'gather' and
                    a.loop.outs == [arg + ' i' for arg in summed.args]), 'bucket cells').loop
        number = int(re.search(r'\d+$', summed.child)[0])
        cell_number = int(re.search(r'\d+', summed.args[0])[0])
        scope = module.scopes[f'parallel_generatedRoot_{cell_number}']
        members = summed.count
        width, remainder = divmod(cell.count, members)
        require(not remainder and width >= 2, 'unsupported bucket width')
        labels = family_count(module.root.wires[cell.args[1].removesuffix('_gathered')].type)
        source = templates[0 if index == 0 else 1]
        old_cell, old_sum, old_index = (148, 149, 139) if index == 0 else (152, 153, 138)
        mapping = {16: cell.args[1].removesuffix('_gathered'),
                   27: cell.args[3].removesuffix('_gathered'),
                   47: cell.args[7].removesuffix('_gathered'),
                   14: cell.args[2], 70: cell.args[11], old_index: cell.args[0].removesuffix(' i')}
        if index == 0:
            mapping.update({140: cell.args[4], 141: cell.args[5], 142: cell.args[6],
                            143: cell.args[8], 144: cell.args[9], 145: cell.args[10],
                            146: cell.args[12].removesuffix(' i'),
                            147: cell.args[13].removesuffix(' i')})
        else:
            mapping.update({149: cell.args[4], 150: cell.args[12].removesuffix(' i'),
                            151: cell.args[13].removesuffix(' i')})
        mapping[old_cell] = f'w_{cell_number}_0'
        mapping[old_sum] = f'w_{number}_0'
        def wire_replace(match):
            prefix, wire, component = match.groups()
            require(int(wire) in mapping, f'unmapped bucket wire {wire}')
            target = mapping[int(wire)]
            return prefix + target.rsplit('_', 1)[0] + '_' + component
        source = re.sub(r'(generatedRoot\.)w_(\d+)_(\d+)', wire_replace, source)
        scope_mapping = {str(old_cell): scope.name, str(old_sum): summed.child}
        source = re.sub(r'parallel_generatedRoot_(\d+)',
                        lambda match: scope_mapping[match[1]], source)
        def fact_replace(match):
            wire = int(match[1])
            require(wire in mapping, f'unmapped bucket fact {wire}')
            current = mapping[wire].rsplit('_', 1)[0] + '_' + match[2]
            return root_fact(module, current)
        source = re.sub(r'F\.c\d+\.h(\d+)_(\d+)', fact_replace, source)
        hashes = [a for a in scope.atoms if a.kind == 'hashSample' and
                  '.decimal' in a.args[2]]
        hashed = hashes[0]
        prefix = f'online1_bucket{index}_rhsPre'
        tag_a, tag_b = re.findall(r'\.decimal \((-?\d+)\)', hashed.args[2])
        result += f'\nabbrev {prefix} : Blob := {hashed.args[1]}\n'
        label_hash = next(a for a in scope.atoms if
                          a.kind == 'hashSample' and '.decimal' not in a.args[2])
        source = source.replace('rhsPre1b' if index else 'rhsPre1', prefix)
        source = source.replace('labelPre', 'online1PrfLabelPre')
        source = source.replace('prfA1', tag_a).replace('prfB1', tag_b)
        source = source.replace(f'online1_bucket{1 if index else 0}', f'online1_bucket{index}', 1)
        source = source.replace('polyNorm (α k) ≤ 1', 'SmulGain (α k) 1')
        source = source.replace('8 * 1 * Bs', 'Bs')
        source = source.replace('PrfCellFlat.rel ', 'PrfCellFlat.rel_mono ')
        source = source.replace('PrfCellFlat.rel_1 ',
                                f'PrfCellFlat.rel_1W_mono (W := {width}) ')
        if index:
            source = source.replace('(by norm_num) hq1 layout',
                                    '(by norm_num) rfl hq1 layout')
        source = source.replace('α : Fin 2 →', f'α : Fin {labels} →')
        source = source.replace('∀ k : Fin 2,', f'∀ k : Fin {labels},')
        source = source.replace('∃ key : Fin 2,', f'∃ key : Fin {labels},')
        source = source.replace('∃ k : Fin 2,', f'∃ k : Fin {labels},')
        source = source.replace('(j : Fin 2)', f'(j : Fin {width})')
        source = source.replace('sel : Fin 8 → Fin 2', f'sel : Fin MEMBERS → Fin {width}')
        source = source.replace('Fin 16', f'Fin {members * width}')
        source = source.replace('Fin 8', 'Fin MEMBERS').replace('(M := 8)', '(M := MEMBERS)')
        source = source.replace(' / 2 < 8 := by', ' / 2 < MEMBERS := by')
        source = source.replace('(W := 2)', f'(W := {width})').replace('(K := 2)', f'(K := {labels})')
        source = source.replace('/ 2', f'/ {width}').replace('2 * ((25 +', f'{width} * ((25 +')
        substitutions = {8: view['ring_dimension'], 25: view['columns'],
                         4096: layout['base'], 2048: layout['base'] // 2}
        source = re.sub(r'\b(8|25|4096|2048)\b', lambda item: str(substitutions[int(item[0])]), source)
        source = source.replace('MEMBERS', str(members))
        if index == 0:
            result += f'abbrev online1PrfLabelPre : Blob := {label_hash.args[1]}\n'
        result += '\nset_option maxHeartbeats 16000000 in\n' + source
    return result


def smudge_prf_chain(package, metadata, source):
    view = metadata['views']['worst']
    n, columns = view['ring_dimension'], view['columns']
    module = Module(package / 'worst', 'W_Stage_online_1', {})
    headers = re.findall(r'theorem online1_bucket\d+ .*? := by', source, re.S)
    require(headers, 'no bucket theorem interfaces')
    depth = len(headers)
    first = headers[0]
    common = first[first.index('{backend :'):first.index('    {ν :')]
    common = re.sub(r'theorem.*', '', common)
    labels = int(re.search(r'α : Fin (\d+)', common)[1])
    members = int(re.search(r'\(sel : Fin (\d+)', first)[1])
    layout = one((item for item in metadata['backend_layouts'] if item['modulus'] == view['Q0']),
                 'chain layout')
    budget_declarations, parameters, proof = '', '', ''
    previous = 'Bs'
    for index, header in enumerate(headers):
        width = int(re.search(r'\(sel : Fin \d+ → Fin (\d+)', header)[1])
        budget = f'''({width} * (({columns} + {columns}) * {n} *
      (({columns} + {columns}) * {n} * Ba * {layout['base'] // 2} + {previous}) *
      {layout['base'] // 2} + {n} * 1 * BB {index}))'''
        budget_declarations += f'''def online1PrfBudget{index} (Ba Bs : Nat) (BB : Fin {depth} → Nat) : Nat :=
  {budget}

'''
        previous = f'online1PrfBudget{index} Ba Bs BB'
        current = 'μ' if index == 0 else f'μ{index - 1} o'
        if index:
            parameters += f'''    (hαμ{index} : ∀ k o, polyNorm (α k * μ{index - 1} o) ≤ 1)
'''
        tail = header[header.index('    {ν :'):header.index('    (o :')]
        tail = re.sub(r' \{hT :.*?\}', '', tail)
        tail = tail.replace(' {BB : Nat}', '')
        for old, new in [('ν', f'ν{index}'), ('eB', f'eB{index}'), ('hrhs', f'hrhs{index}'),
                         ('heB', f'heB{index}'), ('sel', f'sel{index}'), ('hsel', f'hsel{index}')]:
            tail = re.sub(r'\b' + old + r'\b', new, tail)
        tail = re.sub(r'\bBB\b', f'(BB {index})', tail).replace('• hT', '• g')
        index_wire = re.search(r'= generatedRoot\.(w_\d+_\d+) ctx \(cellAt', header)[1]
        parameters += tail + f'''    (selected{index} : Fin {members} → Fin {labels})
    (hselected{index} : ∀ o, ((selected{index} o).val : Int) =
      generatedRoot.{index_wire} ctx (cellAt o (sel{index} o)))
    (μ{index} : Fin {members} → ErrorPoly {n})
    (hproduct{index} : ∀ o, μ{index} o = α (selected{index} o) * ({current}))
'''
        output = header[header.index('      EncRel sm '):header.index('        (fun i ↦')]
        target = f'''(fun port ↦ reducePoly modulus_0 {n} (μ{index} o * unitPair sp port) • g)'''
        proof += f'''  have bucket{index} : ∀ o : Fin {members},
{output}        {target} ({previous}) := by
    intro o
    obtain ⟨key, hkey, hrel⟩ := online1_bucket{index} ctx F hq1 layout hlayout hexact
      {'hα hαμ' if index == 0 else f'hα hαμ{index}'} hG ha hpay hg' hg
      {'hs' if index == 0 else f'(fun member ↦ by simpa only [Nat.cast_one, Int.cast_one, one_mul] using bucket{index - 1} member)'}
      (hrhs{index}) heB{index} sel{index} hsel{index} o
    have same : key = selected{index} o := Fin.ext (by
      exact_mod_cast hkey.trans (hselected{index} o).symm)
    subst key
    simpa only [Nat.cast_one, Int.cast_one, one_mul, ← hproduct{index} o] using hrel
'''
    final = output.replace('      EncRel sm ', '    EncRel sm ')
    return budget_declarations + f'''set_option maxHeartbeats 400000000 in
/-- Every exported PRF bucket is derived from label, seed-state and selector-RHS premises.
Selected-label products are public semantics, not assumed intermediate encodings. -/
theorem online1_prf_root {common}    (BB : Fin {depth} → Nat)
{parameters}    (o : Fin {members}) :
{final}      {target} ({previous}) := by
{proof}  exact bucket{depth - 1} o

'''


def smudge_readout(package, metadata):
    module = Module(package / 'worst', 'W_Stage_online_1', {})
    view = metadata['views']['worst']
    values, scope_name = one(((table['values'], use['scope'])
                             for table in view['lut_tables']['online_1']
                             for use in table['uses'] if
                             use['scope'].startswith('parallel_generatedRoot_')),
                            'step-1 readout')
    branches = len(values)
    require(all(value > 0 and value % 2 == 1 for value in values),
            'readout automorphisms must be positive and odd')
    loop = one((a.loop for a in module.root.loops if a.loop.child == scope_name),
               'readout loop')
    scope = module.scopes[scope_name]
    members = loop.count
    layout = one((item for item in metadata['backend_layouts'] if
                  item['modulus'] == view['Q0']), 'step-1 layout')
    old = (HERE.parent / 'average-n8-p5x50/proof/Dio2Online1Prf.lean').read_text()
    header = old[old.index('theorem online1_smudge_total'):old.index('    EncRel sm (generatedRoot.w_174')]
    mem = old[old.index('  have hmem : ∀ o : Fin 8, '):old.index('  -- Placed at')]
    mem = mem.replace('  have hmem : ∀ o : Fin 8, ', '')
    mem = mem.replace('    intro o\n', '', 1)
    source = header.replace('online1_smudge_total', 'online1_readout_graph', 1)
    source = source.removesuffix(' :\n') + '\n    (o : Fin MEMBERS) :\n    ' + mem
    mapping = {132: loop.args[0], 133: loop.args[1], 134: loop.args[2],
               135: loop.args[3], 136: loop.args[4], 14: loop.args[5],
               137: loop.args[6], 153: loop.args[8].split()[0]}
    source = re.sub(r'generatedRoot\.w_(\d+)_(\d+)',
                    lambda match: 'generatedRoot.' +
                    mapping.get(int(match[1]), loop.outs[0].split()[0]).rsplit('_', 1)[0]
                    + '_' + match[2], source)
    source = source.replace('parallel_generatedRoot_154', scope_name)
    source = source.replace('F.c6.h154_5', root_fact(module, loop.outs[-1].split()[0]))
    source = source.replace('Fin 8', 'Fin MEMBERS')
    source = source.replace('hhelp : ∀ i : Fin MEMBERS', 'hhelp : ∀ i : Fin BRANCHES')
    source = source.replace('C : Fin MEMBERS', 'C : Fin BRANCHES')
    source = source.replace('hswitch : ∀ i : Fin MEMBERS', 'hswitch : ∀ i : Fin BRANCHES')
    source = source.replace('sigmas8_odd', 'online1Sigmas_odd').replace('sigmas8', 'online1Sigmas')
    source = source.replace('helperPre', 'online1PrfHelperPre')
    source = source.replace('(L := 7)', '(L := LAST)').replace('7 * (', 'LAST * (')
    source = source.replace('+ 8 * (8 * 1 * BB)', '+ BRANCHES * (8 * 1 * BB)')
    n = view['ring_dimension']
    source = source.replace('(log := 3)', f'(log := {n.bit_length()-1})')
    substitutions = {8: n, 25: view['columns'], 4096: layout['base'], 2048: layout['base']//2}
    source = re.sub(r'\b(8|25|4096|2048)\b', lambda item: str(substitutions[int(item[0])]), source)
    source = source.replace('MEMBERS', str(members)).replace('BRANCHES', str(branches))
    source = source.replace('LAST', str(branches-1))
    helper_prefix = one({a.args[1] for a in scope.atoms if a.kind == 'hashSample'},
                        'readout helper prefix')
    return f'''
abbrev online1PrfHelperPre : Blob := {helper_prefix}
abbrev online1Sigmas : List Int := {values}

theorem online1Sigmas_odd (i : Fin {branches}) : Odd (sigmaOf online1Sigmas i).toNat := by
  have hsize : online1Sigmas.length = {branches} := rfl
  have hindex : i.val < online1Sigmas.length := by simpa only [hsize] using i.isLt
  have hall : online1Sigmas.all (fun value ↦ value.toNat % 2 == 1) = true := by decide
  have hmod := List.all_eq_true.mp hall _ (List.getElem_mem hindex)
  change Odd ((online1Sigmas[i.val]?.getD 0).toNat)
  rw [List.getElem?_eq_getElem hindex, Option.getD_some]
  exact Nat.odd_iff.mpr (by simpa only [beq_iff_eq] using hmod)

set_option maxHeartbeats 32000000 in
{source}
'''


def smudge_placement_and_pack(package, metadata):
    import gen_online0_chains as shared
    module = Module(package / 'worst', 'W_Stage_online_1', {})
    placement = one((a for a in module.root.loops if a.loop.kind == 'par' and
                     a.loop.child in module.scopes and
                     any('multiplyMonomial' in (wire.term or '')
                         for wire in module.scopes[a.loop.child].wires.values())), 'smudge placement')
    monomial = shared.declaration((HERE.parent / 'average-n8-p5x50/proof/Dio2Online0Sigma.lean').read_text(),
                                  'monomial_place_rel')
    monomial = monomial.replace('monomial_place_rel', 'online1_monomial_place_rel')
    generic = shared.GATHER_PLACEMENT.split('/-- A generated row-sum scope')[0]
    generic = generic.replace(' monomial_place_rel', ' online1_monomial_place_rel')
    generic = generic.replace('tuple_ext', 'online1_tuple_ext')
    generic = generic.replace('gather_monomial_place_rel', 'online1_gather_monomial_place_rel')
    generic = generic.replace('∃ member : Fin members, EncRel',
                              '∃ member : Fin members, member.val = index.toNat ∧ EncRel')
    generic = generic.replace('⟨member, online1_monomial_place_rel',
                              '⟨member, rfl, online1_monomial_place_rel')
    specific = shared.placement_from_scope(module.scopes[placement.loop.child])
    specific = specific.replace('gather_monomial_place_rel', 'online1_gather_monomial_place_rel')
    scope = module.scopes[placement.loop.child]
    get = one({a.args[1] for a in scope.atoms if a.kind == 'familyGetDynamic'}, 'placement index')
    index = shared.adapters.wire_term(scope, get)
    specific = specific.replace('∃ member : Fin ', '∃ member : Fin ', 1)
    specific = re.sub(r'(∃ member : Fin \d+), EncRel',
                       lambda match: match[1] + f', member.val = ({index}).toNat ∧ EncRel', specific)
    n = metadata['views']['worst']['ring_dimension']
    count = placement.loop.count
    packed, remainder = divmod(count, n)
    require(not remainder, 'placement packing is not coefficient-aligned')
    if packed == 1:
        total = shared.member_total_from_graph(module, {'members': count})
        total = total.replace('online0_sigma_total_graph', 'online1_smudge_pack')
    else:
        total = shared.columns_from_graph(module, {'members': count, 'packed_members': packed},
                                          'online1_smudge_pack', n)
    total = total.replace('.g2_sem', '.sem')
    return (monomial + generic + '\nend Dio2Proof\nnamespace W_Stage_online_1\n'
            'open Dio2Proof Mxx.Primitives MxxRuntime MxxExponentLut\n'
            '  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenIface\n' +
            specific + '\nend W_Stage_online_1\nnamespace Dio2Proof\n'
            'open W_Stage_online_1 Mxx.Primitives MxxRuntime MxxExponentLut\n'
            '  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenIface\n'
            'open scoped BigOperators\n' + total)


def final_parenthesized(source):
    stop = source.rfind(')')
    depth = 1
    start = stop - 1
    while depth:
        if source[start] == ')':
            depth += 1
        elif source[start] == '(':
            depth -= 1
        start -= 1
    return source[start + 1:stop + 1]


def smudge_semantic_assembly(package, metadata, chain, readout):
    module = Module(package / 'worst', 'W_Stage_online_1', {})
    view = metadata['views']['worst']
    n, columns, count = view['ring_dimension'], view['columns'], view['smudge_digits']
    placement = one((atom.loop for atom in module.root.loops if atom.loop.kind == 'par' and
                     any('multiplyMonomial' in (wire.term or '') for wire in
                         module.scopes[atom.loop.child].wires.values())), 'semantic placement')
    members = placement.count
    chain_header = chain[chain.index('theorem online1_prf_root '):].split(' := by', 1)[0]
    depth = len(re.findall(r'\(hproduct\d+ :', chain_header))
    prefix = chain_header[len('theorem online1_prf_root '):chain_header.index('    (o :')]
    chain_args = 'ctx F hq1 layout hlayout hexact hα hαμ hG ha hpay hg\' hg hs BB'
    for index in range(depth):
        if index:
            chain_args += f' hαμ{index}'
        chain_args += f' hrhs{index} heB{index} sel{index} hsel{index} selected{index} hselected{index} μ{index} hproduct{index}'
    readout_header = readout[readout.index('theorem online1_readout_graph '):].split(' := by', 1)[0]
    extra = readout_header[readout_header.index('    {Bh :'):readout_header.index('    (o :')]
    extra = re.sub(r'\bμ\b', f'μ{depth - 1}', extra)
    extra = re.sub(r'\bν\b', 'ρ', extra)
    extra = re.sub(r'\bBB\b', 'Bswitch', extra)
    header = prefix + f'    (hsm : polyNorm sm ≤ 1)\n' + extra
    noise = final_parenthesized(readout_header)
    noise = re.sub(r'\bB\b', f'(online1PrfBudget{depth - 1} Ba Bs BB)', noise)
    noise = re.sub(r'\bBB\b', 'Bswitch', noise)
    noise = noise.replace('Ba Bs Bswitch', 'Ba Bs BB')
    budget = f'online1ReadoutBudget Ba Bs Bh Bswitch BB'
    goal = readout_header[readout_header.index('    EncRel sm '):]
    goal = re.sub(r'\bν\b', 'ρ', goal)
    goal = goal[:goal.rfind(final_parenthesized(goal))] + '(' + budget + ')'
    text = f'''def online1ReadoutBudget (Ba Bs Bh Bswitch : Nat) (BB : Fin {depth} → Nat) : Nat :=
  {noise}

set_option maxHeartbeats 400000000 in
/-- The smudge readout follows from the complete exported PRF chain and preprocessing facts. -/
theorem online1_readout_root {header}    (o : Fin {members}) :
{goal} := by
  exact online1_readout_graph ctx F hq1 layout hlayout hexact hsm hg' hg
    (fun member ↦ by simpa only [Nat.cast_one, Int.cast_one, one_mul] using online1_prf_root {chain_args} member)
    hhelp hswitch hμ hwalsh o

/-- The export transposes coefficient-major members into digit-major readout indices. -/
def online1PlacementSource (member : Fin {members}) : Fin {members} :=
  ⟨member.val / {n} + {count} * (member.val % {n}), by have := member.isLt; omega⟩

/-- Each digit's plaintext is the actual corrected placement and column sum. -/
noncomputable def online1PackedReadout (Δ : ErrorPoly {n}) (ρ : Fin {members} → ErrorPoly {n}) :
    Fin {count} → Fin 1 → ErrorPoly {n} := fun digit _ ↦
  ∑ coefficient : Fin {n}, Xe (n := {n}) coefficient.val *
    (Δ * ρ (online1PlacementSource ⟨{n} * digit.val + coefficient.val,
      by have := digit.isLt; have := coefficient.isLt; omega⟩))

set_option maxHeartbeats 400000000 in
/-- Actual placement and packed totals preserve the readout relation without member enumeration. -/
theorem online1_pack_readout {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (ctx : generatedRoot.Ctx backend hashModel params) {{outputs : _}}
    (F : generatedRoot.Facts ctx outputs) {{sm sp Δ : ErrorPoly {n}}}
    (ρ : Fin {members} → ErrorPoly {n}) (gq : ExactMatrix modulus_0 {n} 1 {columns}) {{B : Nat}}
    (hread : ∀ member, EncRel sm
      {tuple_of([f'(generatedRoot.{wire.removesuffix(" i")} ctx member)' for wire in placement.args])}
      (fun port ↦ reducePoly modulus_0 {n} (Δ * ρ member * unitPair sp port) • gq) B)
    (digit : Fin {count}) :
'''
    digit_loop = one((atom.loop for atom in module.root.loops if atom.loop.kind == 'seq' and
                      any(inner.kind == 'blockDiagonalRuns' for inner in module.scopes[atom.loop.child].atoms)),
                     'semantic digit loop')
    packed_wire = digit_loop.args[5]
    packed_prefix = packed_wire.rsplit('_', 1)[0]
    packed = tuple_of([f'(generatedRoot.{packed_prefix}_{component} ctx)' for component in range(6)])
    text += f'''    EncRel sm (blkT (C := {count}) (m := {columns}) {packed} digit)
      (fun port ↦ reducePoly modulus_0 {n}
        (online1PackedReadout Δ ρ digit 0 * unitPair sp port) • gq) ({n} * B) := by
  let target : Fin {members} → Fin 2 → ExactMatrix modulus_0 {n} 1 {columns} := fun member port ↦
    reducePoly modulus_0 {n} (Xe (n := {n}) (member.val % {n}) *
      (Δ * ρ (online1PlacementSource member) * unitPair sp port)) • gq
  have hplace : ∀ member : Fin {members}, EncRel sm
      {tuple_of([f'(generatedRoot.{wire.removesuffix(" i")} ctx member)' for wire in placement.outs])}
      (target member) B := by
    intro member
    obtain ⟨source, hsource, hrel⟩ := {placement.child}.placement_rel
      ({root_fact(module, placement.outs[-1].removesuffix(' i'))} member) hread
    have same : source = online1PlacementSource member := by
      apply Fin.ext
      rw [hsource]
      have hdiv := (Int.ofNat_fdiv member.val {n}).symm
      have hmod := fmod_ofNat member.val {n}
      norm_num only [Nat.cast_ofNat] at hdiv hmod
      simp only [Int.ofNat_eq_natCast] at hmod ⊢
      rw [hdiv, hmod]
      simp only [online1PlacementSource, ← Int.natCast_mul, ← Int.natCast_add,
        Int.toNat_natCast, Nat.one_mul] <;> omega
    subst source
    simpa only [target, smul_smul, ← map_mul] using hrel
'''
    if count == 1:
        text += f'''  have htotal := online1_smudge_pack ctx F hplace
  have hdigit : digit = 0 := Subsingleton.elim _ _
  subst digit
  have heq : blkT (C := 1) (m := {columns}) {packed} 0 = {packed} := by
    apply online1_tuple_ext
    intro component
    rw [blkT_get]
    funext row column
    simp [blk]
  rw [heq]
'''
    else:
        text += '  have htotal := online1_smudge_pack ctx F hplace digit\n'
    text += f'''  apply htotal.target_eq
  funext port
  simp only [target, online1PackedReadout, online1PlacementSource, Finset.sum_apply,
    Finset.sum_smul, map_sum, Finset.sum_mul, map_mul, Fin.val_zero, Nat.zero_mul, Nat.zero_add]
  apply Finset.sum_congr rfl
  intro coefficient _
  simp only [Nat.add_mod, Nat.mul_mod_right, Nat.zero_add, Nat.mod_eq_of_lt coefficient.isLt]
  apply congrArg (fun scalar : ExactPoly modulus_0 {n} ↦ scalar • gq)
  ring

set_option maxHeartbeats 400000000 in
/-- The full root Facts discharge the packed PRF invariant used by the smudge digit loop. -/
theorem online1_packed_prf_root {header}    (digit : Fin {count}) :
    EncRel sm (blkT (C := {count}) (m := {columns}) {packed} digit)
      (fun port ↦ reducePoly modulus_0 {n}
        (online1PackedReadout Δ' ρ digit 0 * unitPair sp port) • gq)
      ({n} * ({budget})) := by
  exact online1_pack_readout ctx F ρ gq
    (fun member ↦ online1_readout_root {chain_args} hsm hhelp hswitch hμ hwalsh member) digit

'''
    return text, header, chain_args + ' hsm hhelp hswitch hμ hwalsh', budget


def output_adapter(package, metadata):
    v = metadata['views']['worst']
    m = Module(package / 'worst', 'W_Stage_online_1', {})
    s, flow = m.root, Flow(m.root)
    switch = one((a for a in s.atoms if a.kind == 'modulusSwitchRuns'), 'final switch')
    zeta, result = switch.args
    require(result in output_wires(s), 'final switch must be an exported output')
    before_half, half = flow.op(zeta, 'matrixSub')
    reduced, total = flow.op(before_half, 'matrixSub')
    state, smudge = flow.op(total, 'matrixAdd')
    product = flow.atom(reduced, 'modulusReduceRuns')[0]
    b, preimage = flow.op(product, 'matrixMul')
    rd = m.dimension(s, result)
    bd = m.dimension(s, b)
    Q, n, h, width, q = v['Q0'], v['ring_dimension'], bd['columns'], rd['columns'], rd['modulus']
    require(m.dimension(s, product)['modulus'] == Q, 'output hint modulus changed')
    require(m.dimension(s, reduced)['modulus'] == Q, 'output reduction changes modulus')
    require(q == v['primes'][0] and Q % q == 0, 'unexpected final prime')
    quotient = Q // q
    def w(wire):
        return f'(generatedRoot.{wire} ctx)'
    def fact(atom):
        wire = atom.args[-1]
        return f'F.c{atom.frag}.h{wire[2:]}'
    return f'''{MARKER}
open W_Stage_online_1 in
/-- The actual exported final switch returns the residues of Y. State, preimage and
smudge invariants are explicit inputs; root subtraction and switching are proved from Facts. -/
theorem online1_output {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (ctx : generatedRoot.Ctx backend hashModel params) {{outputs : _}}
    (F : generatedRoot.Facts ctx outputs)
    {{sm : ErrorPoly {n}}}
    {{Bk : Fin 2 → ExactMatrix {Q} {n} 1 {h}}}
    {{eb : ErrorMatrix {n} 1 {h}}} {{Beb : Nat}}
    (hb : {w(b)} = maskPart (reducePoly {Q} {n} sm) Bk +
      reduceMatrix {Q} {n} 1 {h} eb) (heb : CoeffBound eb Beb)
    {{Pd : ErrorMatrix {n} {h} {width}}} {{BP : Nat}}
    (hP : {w(preimage)} = reduceMatrix {Q} {n} {h} {width} Pd) (hPd : CoeffBound Pd BP)
    {{Kx Ks : Fin 2 → ExactMatrix {Q} {n} 1 {width}}}
    (hpre : ∀ k, Bk k * {w(preimage)} = Kx k + Ks k)
    {{Y : ExactMatrix {Q} {n} 1 {width}}} {{ex : ErrorMatrix {n} 1 {width}}} {{Bx : Nat}}
    (hx : {w(state)} = maskPart (reducePoly {Q} {n} sm) Kx -
      (({quotient} : Int) : ExactPoly {Q} {n}) • Y + reduceMatrix {Q} {n} 1 {width} ex)
    (hex : CoeffBound ex Bx)
    {{T : ExactMatrix {Q} {n} 1 {width}}} {{es : ErrorMatrix {n} 1 {width}}} {{Bs : Nat}}
    (hs : {w(smudge)} = maskPart (reducePoly {Q} {n} sm) Ks - T +
      reduceMatrix {Q} {n} 1 {width} es) (hes : CoeffBound es Bs)
    {{M : ErrorMatrix {n} 1 {width}}} {{BM : Nat}}
    (hTH : T - {w(half)} = reduceMatrix {Q} {n} 1 {width} M) (hM : CoeffBound M BM)
    (hnoise : 2 * ({h} * {n} * Beb * BP + Bx + Bs + BM) < {quotient})
    (c : Fin {width}) (i : Fin {n}) :
    ({w(result)} 0 c).coeff i = ((((Y 0 c).coeff i).val : Nat) : ZMod {q}) := by
  have hz : {w(zeta)} = matrixSub (matrixSub (matrixMul {w(b)} {w(preimage)})
      (matrixAdd {w(smudge)} {w(state)})) {w(half)} := by
    change matrixSub (matrixSub {w(reduced)} (matrixAdd {w(state)} {w(smudge)})) {w(half)} = _
    rw [modulusReduce_same (by norm_num) {fact(flow.producers[reduced])}]
    simp only [matrixAdd]
    rw [add_comm]
    rfl
  exact out_gen (by norm_num) (by norm_num : {Q} = {quotient} * {q})
    (by norm_num) (by norm_num) hb heb hP hPd hpre hx hex hs hes hTH hM hnoise hz
    {fact(switch)} c i
'''


def state_lut_adapter(package, metadata):
    """Instantiate the fixed four-branch state circuit, never the PRF/member families.

    The exported state circuit is shared by the two rows. Refuse a changed scope
    or missing root wires rather than silently emitting a different program proof.
    Kernel checking additionally validates every Facts projection.
    """
    v = metadata['views']['worst']
    m = Module(package / 'worst', 'W_Stage_online_1', {})
    tables = [(t['values'], u['scope']) for t in v['lut_tables']['online_1']
              for u in t['uses'] if len(t['values']) == 4 and
              u['scope'].startswith('sequential_generatedRoot_')]
    sigmas, scope = one(tables, 'four-branch state LUT')
    require(scope == 'sequential_generatedRoot_91', 'state LUT root layout changed')
    old = (HERE.parent / 'average-n8-p5x50/proof/Dio2Online1State.lean').read_text()
    input_proof = old[old.index('theorem online1_lut_input'):old.index('/-- The automorphisms')]
    input_proof = input_proof.replace('cell_flat ', 'cell_flat_mono ')
    input_proof = input_proof.replace('(hα : polyNorm (Xe (n := 8) 2 * α) ≤ 1)',
                                      '(hα : SmulGain (Xe (n := 8) 2 * α) 1)')
    input_proof = input_proof.replace('8 * 1 * Bs', 'Bs')
    from dio2_meta import family_count
    count = family_count(m.root.wires['w_16_0'].type)
    input_proof = input_proof.replace('{f : Fin 2 →', f'{{f : Fin {count} →')
    tags = {}
    for wire, name in [('w_20_0', 'labelPre'), ('w_39_0', 'statePre'), ('w_71_0', 'rhsPreY')]:
        atom = one((a for a in m.root.atoms if a.kind == 'hashSample' and a.args[-1] == wire),
                   name + ' hash')
        tags[name] = atom.args[1]
        if name == 'rhsPreY':
            pa, pb = re.findall(r'\.decimal \((-?\d+)\)', atom.args[2])
    tags_text = ''.join(f'abbrev online1_{k} : Blob := {x}\n' for k,x in tags.items())
    for name in tags:
        input_proof = input_proof.replace(name, 'online1_' + name)
    input_proof = input_proof.replace('prfA1', pa).replace('prfB1', pb)
    source = old[old.index('abbrev sigmas4'):]
    source = source[:source.index('set_option maxHeartbeats 32000000')]
    source = tags_text + '\nset_option maxHeartbeats 32000000 in\n' + input_proof + source
    for wire in set(re.findall(r'generatedRoot\.(w_\d+_\d+)', source)):
        require(wire in m.root.wires, f'missing state wire {wire}')
    layout, = [l for l in metadata['backend_layouts'] if l['modulus'] == v['Q0']]
    n = v['ring_dimension']
    require(n > 0 and n & (n - 1) == 0, 'ring dimension is not a power of two')
    source = source.replace('[1, 3, 5, 15]', str(sigmas))
    source = source.replace('(log := 3)', f'(log := {n.bit_length() - 1})')
    substitutions = {8: n, 25: v['columns'], 4096: layout['base'], 2048: layout['base'] // 2}
    source = re.sub(r'\b(8|25|4096|2048)\b', lambda x: str(substitutions[int(x[0])]), source)
    source = source.replace('helperPre', 'online1HelperPre')
    a = one((a for a in m.root.atoms if a.kind == 'hashSample' and a.args[-1] == 'w_94_0'),
            'last state helper hash')
    prefix = a.args[1]
    return (f'open W_Stage_online_1 RefreshTopProof.GenIface RefreshTopProof.GenInst\n'
            f'abbrev online1HelperPre : Blob := {prefix}\n\n' + source)


def state_key_adapter(package, metadata):
    v = metadata['views']['worst']
    old = (HERE.parent / 'average-n8-p5x50/proof/Dio2KeyAgree1.lean').read_text()
    stack = old[old.index('theorem key1_stack_rows_agree'):old.index('/-- The inputs the online')]
    link = old[old.index('structure Link1'):old.index('  e132 :')]
    proof = old[old.index('theorem state_loop_agree1'):old.index('set_option maxHeartbeats 400000000 in\n/-- The last smudge')]
    text = stack + link + '\nset_option maxHeartbeats 400000000 in\n' + proof
    online = Module(package / 'worst', 'W_Stage_online_1', {})
    keys = Module(package / 'worst', 'W_Stage_keys_online_1_1', {})
    def digit_loop(module):
        return one((atom.loop for atom in module.root.loops if atom.loop.kind == 'seq' and
                    any(inner.kind == 'blockDiagonalRuns' for inner in
                        module.scopes[atom.loop.child].atoms)), 'key digit loop')
    digit_online, digit_keys = digit_loop(online), digit_loop(keys)
    online_final = re.findall(r'w_\d+_\d+', digit_online.final)
    keys_final = re.findall(r'w_\d+_\d+', digit_keys.final)
    state_def = old[old.index('noncomputable def stateKey1'):old.index('/-- The smudge PRF')]
    smudge_def = f'''/-- The actual digit-loop keys, before the final refresh combination. -/
noncomputable def smudgeKeyS1 {{backend : BackendContext}} {{hashModel : HashModel}}
    {{params : W_Stage_online_1.Params}}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) :
    Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 1 :=
    ![W_Stage_online_1.generatedRoot.{online_final[1]} Con,
    W_Stage_online_1.generatedRoot.{online_final[2]} Con]
'''
    theorem = old[old.index('theorem output_key_agree1'):old.index('set_option maxHeartbeats 400000000 in\n/-- A solved')]
    header = theorem[:theorem.index(' := by')]
    header = header.replace('output_key_agree1', 'output_key_agree1_of_smudge')
    header = header.replace('(hexact : layout.droppedModuli = 0) :', f'''(hexact : layout.droppedModuli = 0)
    (hsmudge : ∀ port : Fin 2, smudgeKeyS1 Con port =
      ![W_Stage_keys_online_1_1.generatedRoot.{keys_final[1]} Ck,
        W_Stage_keys_online_1_1.generatedRoot.{keys_final[2]} Ck] port) :''')
    body = theorem[theorem.index('  have a4 :'):theorem.index('  have a76 :')]
    state_end = theorem[theorem.index('  have state0 :'):theorem.index('  have smudgeRow0')]
    def row_slice(module, source, row):
        return one((atom.args[-1] for atom in module.root.atoms if atom.kind == 'sliceMatrix' and
                    atom.args[0] == source and atom.args[1] == str(row)), 'state key row slice')
    def add_wire(module, left, right):
        return one((name for name, wire in module.root.wires.items() if wire.term and
                    re.fullmatch(r'MxxRuntime.matrixAdd ' + left + ' ' + right, wire.term)),
                   'state key sum')
    online0 = row_slice(online, 'w_120_0', 1)
    online1 = row_slice(online, 'w_120_0', 2)
    key1 = row_slice(keys, 'w_116_0', 2)
    late_online = {201: online0, 202: add_wire(online, 'w_91_3', online0),
                   205: online1, 206: add_wire(online, 'w_91_5', online1)}
    late_keys = {179: key1, 180: add_wire(keys, 'w_87_5', key1), 185: output_wires(keys.root)[0]}
    require(all(wire in online.root.wires for wire in late_online.values()), 'state final wires changed')
    require(all(wire in keys.root.wires for wire in late_keys.values()), 'keys final wires changed')
    theorem = header + ''' := by
  obtain ⟨loop2, loop3, loop4, loop5, loop6, loop7, loop8, loop9⟩ :=
    state_loop_agree1 Fon Fk L hq1 layout hlayout hexact
''' + body + state_end
    for module_name, mapping in [('W_Stage_online_1', late_online),
                                 ('W_Stage_keys_online_1_1', late_keys)]:
        theorem = re.sub(module_name + r'\.generatedRoot\.w_(\d+)_0',
                         lambda match: module_name + '.generatedRoot.' +
                         mapping.get(int(match[1]), f'w_{match[1]}_0'), theorem)
    def facts(match):
        context, wire, component = match.groups()
        if int(wire) < 132:
            return match[0]
        module = online if context == 'Fon' else keys
        mapping = late_online if context == 'Fon' else late_keys
        current = mapping.get(int(wire), f'w_{wire}_0').rsplit('_', 1)[0] + '_' + component
        return root_fact(module, current).replace('F.', context + '.')
    theorem = re.sub(r'(Fon|Fk)\.c\d+\.h(\d+)_(\d+)', facts, theorem)
    key_output = output_wires(keys.root)[0]
    key_cat = one((atom for atom in keys.root.atoms if atom.kind == 'concatRows' and
                   atom.args[-1] == key_output), 'output key concatenation')
    key_flow = Flow(keys.root)
    def key_state(port):
        product, noise = key_flow.op(key_cat.args[port], 'matrixAdd')
        state, action = key_flow.op(product, 'matrixMul')
        require(noise == keys_final[port + 1], 'unexpected output smudge key')
        return state, action
    state0, action0 = key_state(0)
    state1, action1 = key_state(1)
    require(action0 == action1, 'different output state actions')
    online_action = re.search(r'have a65\s*:\s*\(?W_Stage_online_1.generatedRoot\.(w_\d+_\d+)', theorem)[1]
    theorem += f'''  have rows := concatRows_two {root_fact(keys, key_output).replace('F.', 'Fk.')}
  intro port
  fin_cases port
  · change rowOf (W_Stage_keys_online_1_1.generatedRoot.{key_output} Ck) 0 = _
    rw [rows.1]
    change W_Stage_keys_online_1_1.generatedRoot.{state0} Ck *
      W_Stage_keys_online_1_1.generatedRoot.{action0} Ck +
      W_Stage_keys_online_1_1.generatedRoot.{keys_final[1]} Ck = _
    rw [← state0, ← a65]
    exact congrArg (fun value ↦ stateKey1 Con 0 *
      W_Stage_online_1.generatedRoot.{online_action} Con + value) (hsmudge 0).symm
  · change rowOf (W_Stage_keys_online_1_1.generatedRoot.{key_output} Ck) 1 = _
    rw [rows.2]
    change W_Stage_keys_online_1_1.generatedRoot.{state1} Ck *
      W_Stage_keys_online_1_1.generatedRoot.{action1} Ck +
      W_Stage_keys_online_1_1.generatedRoot.{keys_final[2]} Ck = _
    rw [← state1, ← a65]
    exact congrArg (fun value ↦ stateKey1 Con 1 *
      W_Stage_online_1.generatedRoot.{online_action} Con + value) (hsmudge 1).symm
'''
    preimage = old[old.index('theorem online1_hpre'):old.index('/-- The published last-step anchor')]
    preimage = preimage.replace('online1_hpre', 'online1_root_hpre_of_smudge')
    preimage = preimage.replace('output_key_agree1 Fon Fk L hq1 layout hlayout hexact k',
                                'output_key_agree1_of_smudge Fon Fk L hq1 layout hlayout hexact hsmudge k')
    preimage = preimage.replace('(hexact : layout.droppedModuli = 0) :', f'''(hexact : layout.droppedModuli = 0)
    (hsmudge : ∀ port : Fin 2, smudgeKeyS1 Con port =
      ![W_Stage_keys_online_1_1.generatedRoot.{keys_final[1]} Ck,
        W_Stage_keys_online_1_1.generatedRoot.{keys_final[2]} Ck] port) :''')
    preimage = preimage.replace('W_Stage_keys_online_1_1.generatedRoot.w_185_0',
                                f'W_Stage_keys_online_1_1.generatedRoot.{key_output}')
    anchor_width = online.dimension(online.root, 'w_1_0')['rows']
    preimage = preimage.replace(' 2 54}', f' 2 {anchor_width}}}')
    text += state_def + smudge_def + '\nset_option maxHeartbeats 400000000 in\n' + theorem + preimage
    text = text.replace('Link1', 'StateLink1').replace('helperPre', 'online1HelperPre')
    sigmas = one((table['values'] for table in v['lut_tables']['online_1']
                  if any(use['scope'] == 'sequential_generatedRoot_91'
                         for use in table['uses'])), 'state key automorphisms')
    text = text.replace('[1, 3, 5, 15]', str(sigmas))
    layout, = [l for l in metadata['backend_layouts'] if l['modulus'] == v['Q0']]
    substitutions = {8: v['ring_dimension'], 25: v['columns'],
                     4096: layout['base'], 2048: layout['base'] // 2}
    text = re.sub(r'\b(8|25|4096|2048)\b', lambda x: str(substitutions[int(x[0])]), text)
    for module in ['W_Stage_online_1', 'W_Stage_keys_online_1_1']:
        m = Module(package / 'worst', module, {})
        for wire in set(re.findall(module + r'\.generatedRoot\.(w_\d+_\d+)', text)):
            require(wire in m.root.wires, f'missing state key wire {module}.{wire}')
    return (MARKER + '\nopen Mxx.ExponentLut.Refresh RefreshTopProof\n'
            '  RefreshTopProof.GenInst RefreshTopProof.GenFinal RefreshTopProof.GenIface\n'
            'variable {n : Nat}\n\n' + text)


def smudge_key_chain(package, metadata):
    import gen_online0_chains as shared
    modules = [Module(package / 'worst', stage, {}) for stage in
               ['W_Stage_online_1', 'W_Stage_keys_online_1_1']]
    view = metadata['views']['worst']
    n, columns = view['ring_dimension'], view['columns']
    def buckets(module):
        result = []
        for atom in module.root.loops:
            loop = atom.loop
            if loop.kind != 'par' or len(loop.args) != 6:
                continue
            cells = [a.loop for a in module.root.loops if a.loop.kind == 'gather' and
                     a.loop.outs == [arg + ' i' for arg in loop.args]]
            if cells:
                result.append((one(cells, 'key cell family'), loop))
        return result
    families = [buckets(module) for module in modules]
    require(len(families[0]) == len(families[1]), 'key PRF depths differ')
    def readout(module):
        return one((atom.loop for atom in module.root.loops if atom.loop.kind == 'par' and
                    any(inner.kind == 'ringAutomorphismRuns' for inner in
                        module.scopes[atom.loop.child].atoms)), 'key readout')
    readouts = [readout(module) for module in modules]
    def root(side, wire):
        wire = wire.removesuffix(' i')
        return f'{modules[side].name}.generatedRoot.{wire} ' + ('Con' if side == 0 else 'Ck')
    def reference(side, wire):
        wire = wire.removesuffix(' i')
        return root_fact(modules[side], wire).replace('F.', 'Fon.' if side == 0 else 'Fk.')
    def loop_tuple(side, loop, member=None):
        wires = re.findall(r'w_\d+_\d+', loop.final) if loop.kind == 'seq' else [
            wire.removesuffix(' i') for wire in loop.outs]
        return tuple_of([root(side, wire) + (f' {member}' if member else '') for wire in wires])
    needed = set(readouts[0].args[:4])
    for cell, _ in families[0]:
        needed.update([cell.args[0].removesuffix(' i'),
                       cell.args[12].removesuffix(' i'), cell.args[13].removesuffix(' i')])
    key_inputs = {wire.term.count('.2'): wire.name for wire in modules[1].root.wires.values()
                  if wire.term and wire.term.startswith('inputs')}
    links = {}
    text = f'''
/-- Shared imported selectors, RHS rows and switching families used by the smudge key circuit.
No ciphertext rows or public-label payloads are equated. -/
structure SmudgeLink1 {{backend : BackendContext}} {{hashModel : HashModel}}
    {{params : W_Stage_online_1.Params}} {{paramsK : W_Stage_keys_online_1_1.Params}}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params)
    (Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK) : Prop where
'''
    for wire in sorted(needed, key=lambda value: int(value.split('_')[1])):
        term = modules[0].root.wires[wire].term
        require(term and term.startswith('inputs'), f'noninput key link {wire}')
        other = key_inputs[term.count('.2') - 2]
        links[wire] = f'S.e_{wire}'
        text += f'  e_{wire} : {root(0, wire)} = {root(1, other)}\n'
    header = '''{backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params} {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
    (S : SmudgeLink1 Con Ck) (hq : 1 < W_Stage_online_1.modulus_0)
    (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 RINGN = some layout)
    (hexact : layout.droppedModuli = 0)'''.replace('RINGN', str(n))
    members = readouts[0].count
    text += f'''
set_option maxHeartbeats 400000000 in
/-- Every exported smudge bucket and the readout compute the same keys in both runs. -/
theorem online1_readout_keys_agree {header} :
    ∀ member : Fin {members}, Enc.A ({loop_tuple(0, readouts[0], 'member')}) =
      Enc.A ({loop_tuple(1, readouts[1], 'member')}) := by
'''
    previous = []
    for index, ((left, summed), (right, summed_right)) in enumerate(zip(*families)):
        if index == 0:
            for port, arg in enumerate([4, 5, 8, 9]):
                left_wire, right_wire = left.args[arg], right.args[arg]
                text += f'''  have hash_input_{port} := {reference(0, left_wire)}
  rw [L.e14] at hash_input_{port}
  have key_input_{port} : {root(0, left_wire)} = {root(1, right_wire)} :=
    (hashSample_eq hash_input_{port}).trans (hashSample_eq {reference(1, right_wire)}).symm
'''
            previous = [f'key_input_{port}' for port in range(4)]
        cells_count = left.count
        width = summed.count and cells_count // summed.count
        require(cells_count == summed.count * width, 'key bucket cell geometry')
        left_number = re.search(r'w_(\d+)_', summed.args[0])[1]
        right_number = re.search(r'w_(\d+)_', summed_right.args[0])[1]
        key_lemma = ('PrfCellFlat.keys (by norm_num)' if index == 0 else
                     f'PrfCellFlat.keys_1W (W := {width}) (by norm_num) rfl')
        left_fact = reference(0, f'w_{left_number}_5')
        right_fact = reference(1, f'w_{right_number}_5')
        rewrites = [links[left.args[0].removesuffix(' i')], 'L.e14'] + previous + [
            'L.e70', links[left.args[12].removesuffix(' i')], links[left.args[13].removesuffix(' i')]]
        cell_left = tuple_of([root(0, f'w_{left_number}_{port}') for port in range(6)])
        cell_right = tuple_of([root(1, f'w_{right_number}_{port}') for port in range(6)])
        side_condition = '' if index == 0 else ' (by have := member.isLt; omega)'
        text += f'''  have cells_{index} : ∀ member : Fin {cells_count},
      Enc.A (Tup.ev (k := 6) ({cell_left}) member) =
        Enc.A (Tup.ev (k := 6) ({cell_right}) member) := by
    intro member
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc⟩ := {left_fact} member
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hd⟩ := {right_fact} member
    refine ({key_lemma} hq layout hlayout hexact
      ({modules[0].name}.parallel_generatedRoot_{left_number}.sem hc){side_condition}).trans
      (Eq.trans ?_ ({key_lemma} hq layout hlayout hexact
        ({modules[1].name}.parallel_generatedRoot_{right_number}.sem hd){side_condition}).symm)
    rw [{', '.join(rewrites)}]
  have bucket_{index} : ∀ member : Fin {summed.count},
      Enc.A ({loop_tuple(0, summed, 'member')}) =
        Enc.A ({loop_tuple(1, summed_right, 'member')}) := by
    intro member
    rw [rowsum_keys ({modules[0].name}.{summed.child}.sem
        ({reference(0, summed.outs[-1])} member)),
      rowsum_keys ({modules[1].name}.{summed_right.child}.sem
        ({reference(1, summed_right.outs[-1])} member))]
    exact Finset.sum_congr rfl fun cell _ ↦ cells_{index} _
'''
        previous = []
        for port, (outer, inner) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            name = f'bucket_{index}_key_{port}'
            previous.append(name)
            text += f'''  have {name} : {root(0, summed.outs[port])} =
      {root(1, summed_right.outs[port])} := funext fun member ↦
    congrFun (congrFun (bucket_{index} member) {outer}) {inner}
'''
    branches = family_count(modules[0].root.wires[readouts[0].args[0]].type)
    children = []
    for module, loop in zip(modules, readouts):
        children.append(one((atom.loop.child for atom in module.scopes[loop.child].loops
                             if atom.loop.kind == 'seq'), 'key readout branch loop'))
    rewrite_readout = [links[wire] for wire in readouts[0].args[:4]] + ['L.e14'] + previous
    text += f'''  intro member
  refine (LutMemberFlat.keys (by norm_num) (L := {branches - 1}) (by norm_num)
    hq layout hlayout hexact
    (fun index current next hbody ↦ {modules[0].name}.{children[0]}.sem hbody)
    ({modules[0].name}.{readouts[0].child}.sem
      ({reference(0, readouts[0].outs[-1])} member))).trans
    (Eq.trans ?_ (LutMemberFlat.keys (by norm_num) (L := {branches - 1}) (by norm_num)
      hq layout hlayout hexact
      (fun index current next hbody ↦ {modules[1].name}.{children[1]}.sem hbody)
      ({modules[1].name}.{readouts[1].child}.sem
        ({reference(1, readouts[1].outs[-1])} member))).symm)
  rw [{', '.join(rewrite_readout)}]
'''
    return text + smudge_key_tail(package, metadata, modules, readouts, header)


def smudge_key_tail(package, metadata, modules, readouts, header):
    import gen_online0_chains as shared
    view = metadata['views']['worst']
    n, columns, count = view['ring_dimension'], view['columns'], view['smudge_digits']
    placements = [one((atom.loop for atom in module.root.loops if atom.loop.kind == 'par' and
                       any('multiplyMonomial' in (wire.term or '') for wire in
                           module.scopes[atom.loop.child].wires.values())), 'key placement')
                  for module in modules]
    text = '''
/-- Gathering is deterministic in its index; monomial placement acts on keys alone. -/
theorem online1_gather_monomial_keys {q n m members : Nat}
    {families : Tup (Fin members → ExactMatrix q n 1 m) 6}
    {gathered output : Tup (ExactMatrix q n 1 m) 6} {index : Int} {exponent : Nat}
    (hgather : GatherUnpack.Spec families index gathered)
    (hplace : ∀ component, output.get component =
      multiplyMonomial (gathered.get component) (exponent : Int)) :
    ∃ member : Fin members, member.val = index.toNat ∧
      Enc.A output = monoKeyF (exponent : Int) (Enc.A (Tup.ev families member)) := by
  obtain ⟨hbound, hnonneg, hgather⟩ := hgather
  let member : Fin members := ⟨index.toNat, hbound⟩
  have hsource : gathered = Tup.ev families member := by
    apply online1_tuple_ext
    intro component
    simpa only [Tup.get_ev] using hgather component
  have hout : output =
      (multiplyMonomial gathered.1 (exponent : Int),
      (multiplyMonomial gathered.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.2.2.1 (exponent : Int), ())))))) := by
    apply online1_tuple_ext
    intro component
    fin_cases component <;> exact hplace _
  refine ⟨member, rfl, ?_⟩
  simpa only [hsource] using monomial_keys hout

'''
    packed_tuples = []
    for side, (module, placement) in enumerate(zip(modules, placements)):
        scope = module.scopes[placement.child]
        binders, call, _ = shared.adapters.header(scope)
        per, atoms = shared.adapters.atom_names(scope)
        gets = [(name, atom) for name, atom in atoms if atom.startswith('MxxRuntime.familyGetDynamic ')]
        gather_outputs = [atom.args[-1] for atom in scope.atoms if atom.kind == 'familyGetDynamic']
        index_wire = one({atom.args[1] for atom in scope.atoms if atom.kind == 'familyGetDynamic'},
                         'key placement index')
        index = shared.adapters.wire_term(scope, index_wire)
        output_name = one((name for name, atom in atoms if atom.startswith('outputs = ')),
                          'key placement output')
        name = 'online1' if side == 0 else 'key1'
        text += f'''open {module.name} in
/-- The actual fused placement exposes only its selected member's keys. -/
theorem {name}_placement_keys {binders} (h : {call}) :
    ∃ member : Fin {readouts[side].count}, member.val = ({index}).toNat ∧
      Enc.A outputs = monoKeyF ((i_0 % {n} : Nat) : Int) (Enc.A (Tup.ev inputs member)) := by
  obtain {shared.adapters.pattern(scope, per, witness=scope.kind == 'record')} := h
  apply online1_gather_monomial_keys
    (families := inputs) (gathered := {tuple_of(gather_outputs)})
    (index := {index}) (exponent := i_0 % {n})
  · apply GatherUnpack.sem (by omega) (by omega)
    intro component
    fin_cases component
    exacts [{', '.join(name for name, _ in gets)}]
  · intro component
    rw [{output_name}]
    have hexponent : Int.fmod (Int.ofNat i_0) ({n} : Int) = ((i_0 % {n} : Nat) : Int) :=
      fmod_ofNat i_0 {n}
    fin_cases component <;> rw [hexponent] <;> rfl

'''
        if count == 1:
            total = shared.member_total_from_graph(module, {'members': placement.count})
            total = re.sub(r'    \{sm : ErrorPoly.*?\(hmember :.*?\n', '', total, flags=re.S)
            total = re.sub(r'EncRel sm (.*?) \(∑ member, target member\) \(\d+ \* budget\)',
                           r'Enc.A \1 = memberKeyF fun member ↦ Enc.A (Tup.ev FAMILIES member)', total)
            total = total.replace('member_total_rel', 'member_total_keys').replace('    hmember\n', '')
            total = total.replace('online0_sigma_total_graph', name + '_packed_keys')
            fams = tuple_of([f'(generatedRoot.{wire} ctx)' for wire in placement.outs
                             for wire in [wire.removesuffix(' i')]])
            total = total.replace('FAMILIES', fams)
            final = one((atom.loop.final for atom in module.root.loops if atom.loop.kind == 'seq' and
                         atom.loop.count == placement.count - 2 and len(atom.loop.args) == 12 and
                         atom.loop.args[-1] == placement.outs[-1].removesuffix(' i')), 'key member sum')
        else:
            total = shared.columns_from_graph(module,
                {'members': placement.count, 'packed_members': count}, name + '_packed_keys', n)
            total = re.sub(r'    \{sm : ErrorPoly.*?\(hmember :.*?\n', '', total, flags=re.S)
            body = total[total.index(' := by'):]
            statement = total[:total.index(' :\n')]
            final = one((atom.loop.final for atom in module.root.loops if atom.loop.kind == 'seq' and
                         atom.loop.count == count - 1 and len(atom.loop.args) == 12), 'key columns')
            final_tuple = tuple_of([f'(generatedRoot.{wire} ctx)' for wire in re.findall(r'w_\d+_\d+', final)])
            fams = tuple_of([f'(generatedRoot.{wire.removesuffix(" i")} ctx)' for wire in placement.outs])
            total = statement + f''' :
    Enc.A (blkT (C := {count}) (m := {columns}) {final_tuple} column) =
      columnKeyF (by norm_num : {n} * {count} ≤ {placement.count})
        (fun member ↦ Enc.A (Tup.ev {fams} member)) column''' + body
            total = total.replace('column_total_rel', 'column_total_keys')
            total = total.replace('    (t := target) (B := budget)\n', '').replace('    hmember column', '    column')
        packed_tuples.append(tuple_of([f'({module.name}.generatedRoot.{wire} ' + ('Con' if side == 0 else 'Ck') + ')'
                                      for wire in re.findall(r'w_\d+_\d+', final)]))
        total = total.replace('.g2_sem', '.sem')
        text += f'open {module.name} in\n' + total
    def wires_tuple(side, loop, member):
        ctx = 'Con' if side == 0 else 'Ck'
        return tuple_of([f'({modules[side].name}.generatedRoot.{wire.removesuffix(" i")} {ctx} {member})'
                         for wire in loop.outs])
    text += f'''set_option maxHeartbeats 400000000 in
/-- Readout equality propagates through corrected gathering and packed column totals. -/
theorem online1_packed_keys_agree {header} (column : Fin {count}) :
    Enc.A (blkT (C := {count}) (m := {columns}) {packed_tuples[0]} column) =
      Enc.A (blkT (C := {count}) (m := {columns}) {packed_tuples[1]} column) := by
  have hread := online1_readout_keys_agree Fon Fk L S hq layout hlayout hexact
  have hplace : ∀ member : Fin {placements[0].count},
      Enc.A ({wires_tuple(0, placements[0], 'member')}) =
        Enc.A ({wires_tuple(1, placements[1], 'member')}) := by
    intro member
    obtain ⟨left, hleft, heqleft⟩ := online1_placement_keys
      ({root_fact(modules[0], placements[0].outs[-1].removesuffix(' i')).replace('F.', 'Fon.')} member)
    obtain ⟨right, hright, heqright⟩ := key1_placement_keys
      ({root_fact(modules[1], placements[1].outs[-1].removesuffix(' i')).replace('F.', 'Fk.')} member)
    have hsame : left = right := Fin.ext (hleft.trans hright.symm)
    rw [heqleft, heqright, hsame]
    exact congrArg (monoKeyF ((member.val % {n} : Nat) : Int)) (hread right)
'''
    if count == 1:
        text += '''  have hcolumn : column = 0 := Subsingleton.elim _ _
  subst column
  have htotal := (online1_packed_keys Con Fon).trans ((congrArg memberKeyF (funext hplace)).trans
      (key1_packed_keys Ck Fk).symm)
  have hblk : ∀ value : Tup (ExactMatrix W_Stage_online_1.modulus_0 RINGN 1 COLS) 6,
      blkT (C := 1) (m := COLS) value 0 = value := by
    intro value
    apply online1_tuple_ext
    intro component
    rw [blkT_get]
    funext row column
    simp [blk]
  rw [hblk, hblk]
  exact htotal
'''
    else:
        text += '''  rw [online1_packed_keys Con Fon column, key1_packed_keys Ck Fk column]
  exact Finset.sum_congr rfl fun coefficient _ ↦ hplace _
'''
    digits = [one((atom.loop for atom in module.root.loops if atom.loop.kind == 'seq' and
                   any(inner.kind == 'blockDiagonalRuns' for inner in module.scopes[atom.loop.child].atoms)),
                  'key digit accumulation') for module in modules]
    final_keys = [re.findall(r'w_\d+_\d+', loop.final)[1:] for loop in digits]
    layout = one((item for item in metadata['backend_layouts'] if item['modulus'] == view['Q0']),
                 'key digit layout')
    digit_count = one((atom.args[2] for atom in modules[0].scopes[digits[0].child].atoms
                       if atom.kind == 'gadgetDecomposeRuns'), 'key gadget count')
    def root(side, wire):
        return f'({modules[side].name}.generatedRoot.{wire} ' + ('Con' if side == 0 else 'Ck') + ')'
    text += f'''set_option maxHeartbeats 400000000 in
/-- The full exported smudge-key branch, including every indexed gadget digit. -/
theorem online1_smudge_keys_agree {header} :
    ∀ port : Fin 2, smudgeKeyS1 Con port =
      ![{root(1, final_keys[1][0])}, {root(1, final_keys[1][1])}] port := by
  have hpacked := online1_packed_keys_agree Fon Fk L S hq layout hlayout hexact
  have hweight : ∀ r : Fin {count}, {root(0, digits[0].args[4])} r =
      {root(1, digits[1].args[4])} r := by
    intro r
    rw [online1_digit_weights Con Fon r, key1_digit_weights Ck Fk r]
  intro port
  fin_cases port
'''
    for port in range(2):
        components = [loop.args[5 + port] for loop in digits]
        terms = [f'''blk {root(side, components[side])} r *
        decG (inner := {columns}) backend {layout['base']} {digit_count}
          ({root(side, digits[side].args[4])} r)''' for side in range(2)]
        text += f'''  · change {root(0, final_keys[0][port])} = {root(1, final_keys[1][port])}
    have hleft := online1_digit_sums Con Fon hq {port + 1}
    have hright := key1_digit_sums Ck Fk hq {port + 1}
    change {root(0, final_keys[0][port])} = ∑ r : Fin {count}, {terms[0]} at hleft
    change {root(1, final_keys[1][port])} = ∑ r : Fin {count}, {terms[1]} at hright
    rw [hleft, hright]
    apply Finset.sum_congr rfl
    intro r _
    have hkey := congrFun (congrFun (hpacked r) {port}) 1
    change blk {root(0, components[0])} r = blk {root(1, components[1])} r at hkey
    rw [hkey, hweight r]
'''
    state = state_key_adapter(package, metadata)
    for old, new in [('output_key_agree1_of_smudge', 'output_key_agree1'),
                     ('online1_root_hpre_of_smudge', 'online1_root_hpre')]:
        statement = state[state.index('theorem ' + old):].split(' := by', 1)[0]
        statement = statement.replace('theorem ' + old, 'theorem ' + new)
        statement = statement.replace('(L : StateLink1 Con Ck)',
                                      '(L : StateLink1 Con Ck) (S : SmudgeLink1 Con Ck)')
        statement = re.sub(r'    \(hsmudge :.*?\] port\) :', '    :', statement, flags=re.S)
        text += f'''\n/-- Root key agreement needs only shared imported inputs and the solved preimage. -/
{statement} := by
  exact {old} Fon Fk L hq1 layout hlayout hexact
    (online1_smudge_keys_agree Fon Fk L S hq1 layout hlayout hexact)
'''
    text = text.replace('RINGN', str(n)).replace('COLS', str(columns))
    return re.sub(r'\bblk ', f'blk (C := {count}) (m := {columns}) ', text)


def smudge_adapter(package, metadata):
    from gen_noise_cert import package_parameters
    v = metadata['views']['worst']
    params = package_parameters(package, metadata)
    Q, n, P, digits = v['Q0'], v['ring_dimension'], params['prf_P'], v['smudge_digits']
    m = Module(package / 'worst', 'W_Stage_online_1', {})
    flow = Flow(m.root)
    switch = one((a for a in m.root.atoms if a.kind == 'modulusSwitchRuns'), 'final switch')
    _, halfwire = flow.op(switch.args[0], 'matrixSub')
    half, = re.findall(r'allCoefficients \((-?\d+)\)', m.root.wires[halfwire].term)
    require(2 * int(half) % Q == 1, 'incorrect exported half offset')
    return f'''{MARKER}
open W_Stage_online_1 in
/-- Multi-digit hTH with the coefficientwise offset actually exported by online step 1.
The readout and weighted-target identities remain the PRF/digit-loop adapter obligations. -/
theorem online1_smudge_hTH {{backend : BackendContext}} {{hashModel : HashModel}}
    {{params : Params}} (ctx : generatedRoot.Ctx backend hashModel params)
    (digit : Fin {digits} → Fin 1 → Fin {n} → Int)
    (readout : Fin {digits} → Fin 1 → ErrorPoly {n})
    (hreadout : ∀ r c, reducePoly {Q} {n} (readout r c) =
      reducePoly {Q} {n} (intPoly (digit r c)))
    (hodd : ∀ r c i, digit r c i % 2 = 1)
    (hbound : ∀ r c i, |digit r c i| ≤ ({P} : Int) - 1)
    {{T : ExactMatrix {Q} {n} 1 1}}
    (hT : T = fun _ c ↦ ∑ r : Fin {digits}, reducePoly {Q} {n} (readout r c) *
      (({half} : ExactPoly {Q} {n}) * ({P} : ExactPoly {Q} {n}) ^ r.val)) :
    ∃ M : ErrorMatrix {n} 1 1, CoeffBound M ({P} ^ {digits} / 2) ∧
      T - generatedRoot.{halfwire} ctx = reduceMatrix {Q} {n} 1 1 M := by
  exact smudge_hTH (d := {digits - 1}) (by norm_num) (by norm_num)
    (half := {half}) (by
      apply (ZMod.intCast_eq_intCast_iff (2 * {half}) 1 {Q}).2
      norm_num [Int.ModEq]) digit readout hreadout hodd hbound hT rfl
'''


def digit_graph(package, metadata, stage='W_Stage_online_1', keys_only=False):
    import gen_online0_chains as shared
    from gen_noise_cert import package_parameters
    module = Module(package / 'worst', stage, {})
    loop = one((atom.loop for atom in module.root.loops if atom.loop.kind == 'seq' and
                any(atom.kind == 'blockDiagonalRuns' for atom in
                    module.scopes[atom.loop.child].atoms)), 'smudge digit loop')
    scope = module.scopes[loop.child]
    count = loop.count
    view = metadata['views']['worst']
    n, columns = view['ring_dimension'], view['columns']
    layout = one((item for item in metadata['backend_layouts'] if item['modulus'] == view['Q0']),
                 'digit gadget layout')
    binders, call, _ = shared.adapters.header(scope)
    per, atoms = shared.adapters.atom_names(scope)
    def names(kind):
        return [name for name, text in atoms if text.startswith('MxxRuntime.' + kind + ' ')]
    slices = names('sliceMatrix')
    require(len(slices) == 3 and len(scope.inputs) == 7 and len(scope.output) == 3,
            'unexpected digit accumulator geometry')
    gather, = names('familyGetDynamic')
    diagonal, = names('blockDiagonalRuns')
    rebase, = names('centeredRebaseRuns')
    decompose, = names('gadgetDecomposeRuns')
    compact, = names('compactCenteredRebaseRuns')
    layout['digits'] = one((atom.args[2] for atom in scope.atoms if atom.kind == 'gadgetDecomposeRuns'),
                            'digit gadget count')
    block = one((atom.args[-1] for atom in scope.atoms if atom.kind == 'blockDiagonalRuns'), 'block')
    source = one((atom.args[-2] for atom in scope.atoms if atom.kind == 'blockDiagonalRuns'), 'weight')
    rebased = one((atom.args[-1] for atom in scope.atoms if atom.kind == 'centeredRebaseRuns'), 'rebase')
    digits_wire = one((atom.args[-1] for atom in scope.atoms if atom.kind == 'compactCenteredRebaseRuns'),
                      'rebased digits')
    sliced = [atom.args[-1] for atom in scope.atoms if atom.kind == 'sliceMatrix']
    output_eq = one((name for name, text in atoms if text.startswith('outputs = ')), 'digit output')
    packed_row, weights, packed0, packed1 = loop.args[3:]
    final_wires = re.findall(r'w_\d+_\d+', loop.final)
    parameters = package_parameters(package, metadata)
    radix = parameters['prf_P']
    half = (view['Q0'] + 1) // 2
    weight_loop = one((atom.loop for atom in module.root.loops if
                       weights in ' '.join(atom.loop.outs)), 'indexed weight family')
    require(weight_loop.kind == 'par' and weight_loop.count == count,
            'weight family is not digit-indexed')
    weight_source, = weight_loop.args
    require(re.findall(r'matrixPolynomial \[\(?(-?\d+)\)?\]',
                       module.root.wires[weight_source].term) == [str(half)],
            'unexpected scalar half source')
    if not keys_only:
        flow = Flow(module.root)
        switch = one((atom for atom in module.root.atoms if atom.kind == 'modulusSwitchRuns'),
                     'digit final switch')
        _, half_offset = flow.op(switch.args[0], 'matrixSub')
    def rw(wire):
        return f'generatedRoot.{wire} ctx'
    packed = [f'inputs.2.2.2.1', 'inputs.2.2.2.2.2.1', 'inputs.2.2.2.2.2.2.1']
    step = f'''
open W_Stage_online_1 RefreshTopProof.GenInst RefreshTopProof.GenIface
/-- One exported digit step uses the digit-indexed weight and the matching packed blocks. -/
theorem online1_digit_step_graph {binders}
    (hq : 1 < modulus_0) (h : {call}) (hr : i_0 < {count}) :
    ∀ component : Fin 3, Tup.get (k := 3) outputs component = inputs.get component.castAdd +
      ![blk {packed[0]} ⟨i_0, hr⟩, blk {packed[1]} ⟨i_0, hr⟩,
        blk {packed[2]} ⟨i_0, hr⟩] component *
        decG (inner := {columns}) backend {layout['base']} {layout['digits']}
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩) := by
  obtain {shared.adapters.pattern(scope, per, witness=False)} := h
  obtain ⟨_, hweight⟩ := familyGet_atG i_0 {gather}
  have hblock : {block} = {source} := by
    funext row column
    simpa only [Fin.isValue, Fin.val_zero, Nat.zero_mul, Nat.zero_add, ite_true]
      using {diagonal}.2 0 0 row column
  have hrebased : {rebased} = {source} :=
    (centeredRebaseRuns_unique {rebase} (centeredRebaseRuns_self hq (by norm_num) _)).trans hblock
  have hscalar : matrixMulScalarLeft {rebased}
      (matrixPolynomial [1]) = inputs.2.2.2.2.1 ⟨i_0, hr⟩ := by
    rw [hrebased, hweight]
    funext row column
    have hrow : row = 0 := Subsingleton.elim _ _
    have hcolumn : column = 0 := Subsingleton.elim _ _
    subst row column
    simp [matrixMulScalarLeft, matrixPolynomial_single]
  have hdigits : {digits_wire} = decG (inner := {columns}) backend {layout['base']} {layout['digits']}
      (inputs.2.2.2.2.1 ⟨i_0, hr⟩) := by
    have hd := (RefreshTopProof.GenFinal.compactRebase_self_gen (by norm_num) hq {compact}).trans
      (decG_eq {decompose})
    simpa only [hscalar] using hd
'''
    for index, (name, wire, original) in enumerate(zip(slices, sliced, packed)):
        step += f'  have hs{index} : {wire} = blk {original} ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ {name} rfl\n'
    step += f'''  rw [{output_eq}]
  intro component
  fin_cases component
'''
    for index, (wire, original, current) in enumerate(zip(sliced, packed,
                                                        ['inputs.1', 'inputs.2.1', 'inputs.2.2.1'])):
        step += f'''  · change matrixAdd {current} (matrixMul {wire} {digits_wire}) =
      {current} + blk {original} ⟨i_0, hr⟩ *
        decG (inner := {columns}) backend {layout['base']} {layout['digits']}
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs{index}, hdigits]
    rfl
'''
    step = step.replace('inputs.get component.castAdd',
                        '![inputs.1, inputs.2.1, inputs.2.2.1] component')
    term = f'''![blk ({rw(packed_row)}) r, blk ({rw(packed0)}) r,
        blk ({rw(packed1)}) r] component *
      decG (inner := {columns}) backend {layout['base']} {layout['digits']} ({rw(weights)} r)'''
    step += f'''
/-- The root smudge accumulator is the sum of all indexed digit actions. -/
theorem online1_digit_sums {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (ctx : generatedRoot.Ctx backend hashModel params) {{outputs : _}}
    (F : generatedRoot.Facts ctx outputs) (hq : 1 < modulus_0) :
    ∀ component : Fin 3, Tup.get (k := 3) ({tuple_of([rw(wire) for wire in final_wires])}) component =
      ∑ r : Fin {count}, {term} := by
  let term : Nat → Fin 3 → ExactMatrix modulus_0 {n} 1 1 := fun index component ↦
    if valid : index < {count} then
      let r : Fin {count} := ⟨index, valid⟩
      {term}
    else 0
  have result := online1_digit_fold term (fun index current next hi hstep ↦ by
    change index < {count} at hi
    have hs := online1_digit_step_graph hq hstep hi
    intro component
    have hcurrent : Tup.get (k := 3) current component =
        ![current.1, current.2.1, current.2.2.1] component := by
      fin_cases component <;> rfl
    dsimp only [term]
    simp only [dif_pos hi, if_pos hi, hcurrent]
    exact hs component) ({root_fact(module, final_wires[-1])})
  intro component
  simpa only [Finset.sum_range, term, dif_pos, Fin.is_lt] using result component
'''
    packed_wires = [packed0.rsplit('_', 1)[0] + '_' + str(component) for component in range(6)]
    packed_tuple = tuple_of([rw(wire) for wire in packed_wires])
    require(packed_row == packed_wires[5] and packed0 == packed_wires[1] and
            packed1 == packed_wires[3], 'unexpected digit key/row component order')
    step += f'''
/-- Every exported digit uses its own radix power, not a repeated first-digit weight. -/
theorem online1_digit_weights {{backend : BackendContext}} {{hashModel : HashModel}} {{params : Params}}
    (ctx : generatedRoot.Ctx backend hashModel params) {{outputs : _}}
    (F : generatedRoot.Facts ctx outputs) (r : Fin {count}) :
    {rw(weights)} r = fun _ _ ↦ ({half} : ExactPoly modulus_0 {n}) *
      ({radix} : ExactPoly modulus_0 {n}) ^ r.val := by
  have equation := ({root_fact(module, weights)} r).2.2
  change {rw(weights)} r = matrixMulScalarLeft
    (liftInteger (({radix} : Int) ^ r.val)) (matrixPolynomial [({half})]) at equation
  rw [equation]
  funext row column
  simp only [matrixMulScalarLeft, liftInteger, matrixPolynomial_single, Int.cast_pow,
    Int.cast_natCast]
  ring
'''
    if keys_only:
        return 'section\n' + step.replace('open W_Stage_online_1 ', f'open {stage} ').replace(
            'online1_', 'key1_').replace('key1_digit_fold', 'online1_digit_fold') + '\nend\n'
    step += f'''

/-- The actual digit loop discharges the smudge row equation from packed readouts.
The packed PRF relation is an explicit composition boundary, not an imported-input fact. -/
theorem online1_smudge_of_packed {{backend : BackendContext}} {{hashModel : HashModel}}
    {{params : Params}} (ctx : generatedRoot.Ctx backend hashModel params) {{outputs : _}}
    (F : generatedRoot.Facts ctx outputs) (hq : 1 < modulus_0)
    (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 {n} = some layout)
    (hexact : layout.droppedModuli = 0)
    {{sm : ErrorPoly {n}}} {{g : ExactMatrix modulus_0 {n} 1 {columns}}}
    (hg : gadgetMatrixRuns backend {layout['base']} {layout['digits']} g)
    (target : Fin {count} → Fin 2 → ExactPoly modulus_0 {n}) {{budget : Nat}}
    (hpacked : ∀ r, EncRel sm (blkT ({packed_tuple}) r)
      (fun port ↦ target r port • g) budget) :
    ∃ error : ErrorMatrix {n} 1 1,
      CoeffBound error ({count} * ({columns} * {n} * budget * {layout['base'] // 2})) ∧
      {rw(final_wires[0])} = Mxx.ExponentLut.Refresh.maskPart (reducePoly modulus_0 {n} sm)
        ![{rw(final_wires[1])}, {rw(final_wires[2])}] -
        (∑ r, target r 1 • {rw(weights)} r) + reduceMatrix modulus_0 {n} 1 1 error := by
  obtain ⟨error, herror, equation⟩ := online1_digit_encoding (by norm_num) hq layout
    hlayout hexact hg ({packed_tuple}) target hpacked ({rw(weights)})
    (fun r ↦ decG (inner := {columns}) backend {layout['base']} {layout['digits']}
      ({rw(weights)} r))
    (fun r ↦ online1_weight_decomposition (by norm_num) hq layout hlayout hexact hg _)
  have row_eq := online1_digit_sums ctx F hq 0
  have key0_eq := online1_digit_sums ctx F hq 1
  have key1_eq := online1_digit_sums ctx F hq 2
  change {rw(final_wires[0])} = ∑ r, blk ({rw(packed_row)}) r *
    decG (inner := {columns}) backend {layout['base']} {layout['digits']} ({rw(weights)} r) at row_eq
  change {rw(final_wires[1])} = ∑ r, blk ({rw(packed0)}) r *
    decG (inner := {columns}) backend {layout['base']} {layout['digits']} ({rw(weights)} r) at key0_eq
  change {rw(final_wires[2])} = ∑ r, blk ({rw(packed1)}) r *
    decG (inner := {columns}) backend {layout['base']} {layout['digits']} ({rw(weights)} r) at key1_eq
  rw [← row_eq, ← key0_eq, ← key1_eq] at equation
  exact ⟨error, herror, equation⟩

/-- The weighted target produced by the digit loop has the multi-digit residual bound.
No internal weighted-target identity is assumed. Odd digit witnesses remain a readout premise. -/
theorem online1_digit_target_hTH {{backend : BackendContext}} {{hashModel : HashModel}}
    {{params : Params}} (ctx : generatedRoot.Ctx backend hashModel params) {{outputs : _}}
    (F : generatedRoot.Facts ctx outputs)
    (digit : Fin {count} → Fin 1 → Fin {n} → Int)
    (readout : Fin {count} → Fin 1 → ErrorPoly {n})
    (hreadout : ∀ r c, reducePoly modulus_0 {n} (readout r c) =
      reducePoly modulus_0 {n} (intPoly (digit r c)))
    (hodd : ∀ r c i, digit r c i % 2 = 1)
    (hbound : ∀ r c i, |digit r c i| ≤ ({radix} : Int) - 1) :
    ∃ error : ErrorMatrix {n} 1 1, CoeffBound error ({radix} ^ {count} / 2) ∧
      (∑ r, reducePoly modulus_0 {n} (readout r 0) • {rw(weights)} r) -
        {rw(half_offset)} = reduceMatrix modulus_0 {n} 1 1 error := by
  apply online1_smudge_hTH ctx digit readout hreadout hodd hbound
  funext row column
  have hcolumn : column = 0 := Subsingleton.elim _ _
  subst column
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
    online1_digit_weights ctx F]
'''
    return step


def smudge_root(package, metadata, header, arguments, budget):
    from gen_noise_cert import package_parameters
    module = Module(package / 'worst', 'W_Stage_online_1', {})
    view = metadata['views']['worst']
    n, columns, count = view['ring_dimension'], view['columns'], view['smudge_digits']
    layout = one((item for item in metadata['backend_layouts'] if item['modulus'] == view['Q0']),
                 'smudge root layout')
    gadget_digits = one((atom.args[2] for scope in module.scopes.values() for atom in scope.atoms
                         if scope.name.startswith('sequential_generatedRoot_') and
                         atom.kind == 'gadgetDecomposeRuns' and
                         any(inner.kind == 'blockDiagonalRuns' for inner in scope.atoms)),
                        'smudge root gadget')
    loop = one((atom.loop for atom in module.root.loops if atom.loop.kind == 'seq' and
                any(inner.kind == 'blockDiagonalRuns' for inner in module.scopes[atom.loop.child].atoms)),
               'smudge root digit loop')
    final = re.findall(r'w_\d+_\d+', loop.final)
    flow = Flow(module.root)
    switch = one((atom for atom in module.root.atoms if atom.kind == 'modulusSwitchRuns'), 'smudge switch')
    _, offset = flow.op(switch.args[0], 'matrixSub')
    radix = package_parameters(package, metadata)['prf_P']
    readout = 'online1PackedReadout Δ\' ρ'
    target = f'fun r port ↦ reducePoly modulus_0 {n} ({readout} r 0 * unitPair sp port)'
    noise = f'{count} * ({columns} * {n} * ({n} * ({budget})) * {layout["base"] // 2})'
    return f'''set_option maxHeartbeats 400000000 in
/-- Full exported step-1 smudge assembly. No internal PRF, packing, digit or target identity is assumed. -/
theorem online1_smudge_root {header}
    (hgq : gadgetMatrixRuns backend {layout['base']} {gadget_digits} gq)
    (digit : Fin {count} → Fin 1 → Fin {n} → Int)
    (hreadout : ∀ r c, reducePoly modulus_0 {n} ({readout} r c) =
      reducePoly modulus_0 {n} (intPoly (digit r c)))
    (hodd : ∀ r c coefficient, digit r c coefficient % 2 = 1)
    (hbound : ∀ r c coefficient, |digit r c coefficient| ≤ ({radix} : Int) - 1) :
    ∃ es : ErrorMatrix {n} 1 1, ∃ M : ErrorMatrix {n} 1 1,
      CoeffBound es ({noise}) ∧ CoeffBound M ({radix} ^ {count} / 2) ∧
      generatedRoot.{final[0]} ctx = Mxx.ExponentLut.Refresh.maskPart (reducePoly modulus_0 {n} sm)
        ![generatedRoot.{final[1]} ctx, generatedRoot.{final[2]} ctx] -
        (∑ r, reducePoly modulus_0 {n} ({readout} r 0) • generatedRoot.{loop.args[4]} ctx r) +
        reduceMatrix modulus_0 {n} 1 1 es ∧
      (∑ r, reducePoly modulus_0 {n} ({readout} r 0) • generatedRoot.{loop.args[4]} ctx r) -
        generatedRoot.{offset} ctx = reduceMatrix modulus_0 {n} 1 1 M := by
  have hpacked := fun r ↦ online1_packed_prf_root {arguments} r
  obtain ⟨es, hes, hs⟩ := online1_smudge_of_packed ctx F hq1 layout hlayout hexact hgq
    ({target}) hpacked
  obtain ⟨M, hM, hTH⟩ := online1_digit_target_hTH ctx F digit ({readout}) hreadout hodd hbound
  refine ⟨es, M, hes, hM, ?_, hTH⟩
  simpa only [unitPair, Matrix.cons_val_one, Matrix.cons_val_zero, mul_one] using hs
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    data = extract(a.package)
    out = a.out or a.package / 'proof'
    texts = {name: (TEMPLATE / f'{name}.lean').read_text() for name in COMMON}
    prf = texts['Dio2Online1Prf'].split(PRF_MARKER)[0].removesuffix('end Dio2Proof\n')
    prf = re.sub(r'(/-!\n# Step-1 PRF composition with arbitrary bucket widths\n).*?(-/)',
        r'''\1
The generic layer retains arbitrary family sizes and signed-monomial action gain one.
Generated root theorems connect every exported bucket, readout, corrected fused
placement and packed total. Imported label/seed-state encodings, selector RHS,
public selected-label products, switching and Walsh facts remain explicit;
intermediate PRF or packing encodings are derived, not assumed.
\2''', prf, count=1, flags=re.S)
    if 'import W_Stage_online_1_Facts' not in prf:
        prf = 'import W_Stage_online_1_Facts\nimport Gen_W_Stage_online_1_Scopes\n' + prf
    prf += '\nattribute [local irreducible] Mxx.Primitives.negacyclicModulus\n\n'
    buckets = prf_buckets(a.package, data)
    chain = smudge_prf_chain(a.package, data, buckets)
    readout = smudge_readout(a.package, data)
    assembly, root_header, root_arguments, root_budget = smudge_semantic_assembly(a.package, data, chain, readout)
    texts['Dio2Online1Prf'] = (prf + buckets + chain + readout +
                             smudge_placement_and_pack(a.package, data) + assembly + '\nend Dio2Proof\n')
    keys = texts['Dio2KeyAgree1'].split(MARKER)[0].removesuffix('end Dio2Proof\n')
    keys = re.sub(r'(/-!\n# Step-1 key agreement composition\n).*?(-/)',
        r'''\1
The generated root proof follows every PRF bucket, LUT, corrected gather/placement,
packed column and indexed gadget digit in the online and zero/key runs.
`StateLink1` and `SmudgeLink1` contain shared imported-input equalities only;
no intermediate ciphertext or key equality is assumed by `output_key_agree1`.
`online1_root_hpre` transports the externally solved preimage equation to these keys.
\2''', keys, count=1, flags=re.S)
    if 'import W_Stage_keys_online_1_1_Facts' not in keys:
        keys = 'import W_Stage_keys_online_1_1_Facts\nimport Gen_W_Stage_keys_online_1_1_Scopes\n' + keys
    if 'import Dio2SmudgeHalf' not in keys:
        keys = 'import Dio2SmudgeHalf\n' + keys
    texts['Dio2KeyAgree1'] = (keys + state_key_adapter(a.package, data) +
                            digit_graph(a.package, data, 'W_Stage_keys_online_1_1', True) +
                            smudge_key_chain(a.package, data) + '\nend Dio2Proof\n')
    half = texts['Dio2SmudgeHalf'].split(MARKER)[0].removesuffix('end Dio2Proof\n')
    half = re.sub(r'(/-!\n# Multi-digit hTH witnesses\n).*?(-/)',
        r'''\1
The generic residual proof applies to every digit, prime and part symbolically.
Generated adapters prove the actual block-diagonal/gadget digit loop, indexed
weights, weighted target and half offset. `online1_smudge_root` composes the full
root PRF/packing proof with this loop and produces both hs and hTH.
Imported encodings, preprocessing semantics and odd bounded readout digits remain
explicit; internal PRF, placement, packing and target identities do not.
\2''', half, count=1, flags=re.S)
    if 'import W_Stage_online_1_Facts' not in half:
        half = 'import W_Stage_online_1_Facts\n' + half
    if 'import Gen_W_Stage_online_1_Scopes' not in half:
        half = 'import Gen_W_Stage_online_1_Scopes\n' + half
    if 'import Dio2Online1Prf' not in half:
        half = 'import Dio2Online1Prf\n' + half
    texts['Dio2SmudgeHalf'] = (half + smudge_adapter(a.package, data) + digit_graph(a.package, data) +
        smudge_root(a.package, data, root_header, root_arguments, root_budget) + '\nend Dio2Proof\n')
    state = (TEMPLATE / 'Dio2Online1State.lean').read_text().split(MARKER)[0]
    state = state.removesuffix('end Dio2Proof\n')
    state = state.replace('alias online1_output := out_gen', 'alias online1_output_gen := out_gen')
    state = state.replace('alias online1_state_lut := LutMemberFlat.rel', 'alias online1_state_lut_gen := LutMemberFlat.rel')
    state = state.replace('alias online1_lut_input := cell_flat_mono', 'alias online1_lut_input_gen := cell_flat_mono')
    if 'import W_Stage_online_1_Facts' not in state:
        state = 'import W_Stage_online_1_Facts\n' + state
    if 'import Gen_W_Stage_online_1_Scopes' not in state:
        state = 'import Gen_W_Stage_online_1_Scopes\n' + state
    texts['Dio2Online1State'] = state + output_adapter(a.package, data) + state_lut_adapter(a.package, data) + '\nend Dio2Proof\n'
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (out / f'{name}.lean').write_text(text)


if __name__ == '__main__':
    main()
