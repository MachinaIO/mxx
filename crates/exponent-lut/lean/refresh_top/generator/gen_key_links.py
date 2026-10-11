"""Write `Part_<c_j>_KeyMask.lean`: the keys of the online root's mask carrier are those the
preprocessing computes for its preimage targets, link by link (cells, member sums, readout LUT,
sampling LUT with the offset, placement, member sum, move down).

The theorem binds the atoms of both roots (the preprocessing's prefixed, see `chain.prefixed`)
and assumes the equalities of the online inputs with the preprocessing's wires they come from.

Usage: `python3 gen_key_links.py <parameter directory>`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import chain_binders, prefixed, seg_args
from gen_part import parts_of
from segment import abbreviations

d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
BODY = open(os.path.join(os.path.dirname(__file__), 'templates', 'KeyMask.body')).read()

for part in parts_of(d):
    kr = open(f'{d}/Part_{part}_KeysRoot.lean').read()
    on_ns, pre_ns = f'Stage_online_eval_{part}', f'Stage_preprocess_{part}'
    on_atoms = ['h115_5', 'h116_5', 'h119_5', 'h120_5', 'h121_5', 'h312_5', 'h313_5'] + \
        [f'h{n}_0' for n in (314, 315, 317, 318, 320, 321, 323, 324, 326, 327, 329, 330)] + ['h332_5']
    ob = chain_binders(f'{d}/generated/{on_ns}.lean', on_ns,
                       [(kr, 'keyRootOffset'), (kr, 'keyRootLut186'), (kr, 'keyRootLut252')], ab, on_atoms)
    pre_atoms = ['h35_7', 'h36_3', 'h39_7', 'h40_3', 'h42_3', 'h146_3', 'h147_0', 'h148_0', 'h149_0',
                 'h150_0', 'h151_3', 'h152_3', 'h153_3'] + \
        [f'h{n}_0' for n in (154, 155, 157, 158, 160, 161, 163, 164)] + ['h166_3', 'h167_0', 'h338_0']
    pb = [prefixed(b) for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns,
                                             [(kr, 'preKeyOffset')], ab, pre_atoms)]
    call = lambda th, p='': ' '.join(f'({a} := {p}{a})' for a in seg_args(kr, th))
    body = BODY.replace('@P@', part).replace('@ON@', on_ns).replace('@PRE@', pre_ns)
    for th in ['keyRootOffset', 'keyRootLut186', 'keyRootLut252']:
        body = body.replace(f'@ARGS({th})@', call(th))
    body = body.replace('@ARGS(preKeyOffset)@', call('preKeyOffset', 'p'))
    body = body.replace('@BINDERS@', '\n'.join('    ' + b for b in ob + pb))
    open(f'{d}/Part_{part}_KeyMask.lean', 'w').write(body)
    # The source LUT.
    sb = chain_binders(f'{d}/generated/{on_ns}.lean', on_ns, [(kr, 'keyRootLut18')], ab)
    pb = [prefixed(b) for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab,
                                             ['h185_3', 'h186_0', 'h342_0'])]
    body = open(os.path.join(os.path.dirname(__file__), 'templates', 'KeySource.body')).read()
    row = int(part[2])
    body = body.replace('@KX0@', ['w_52_0', 'w_67_0'][row]).replace('@KX1@', ['w_70_0', 'w_73_0'][row])
    body = body.replace('@KXP@', ['⟨a.1, a.2.2.1⟩', '⟨a.2.1, a.2.2.2.1⟩'][row]).replace('@ROW@', str(row))
    body = body.replace('@P@', part).replace('@ON@', on_ns).replace('@PRE@', pre_ns)
    body = body.replace('@ARGS(keyRootLut18)@', call('keyRootLut18'))
    body = body.replace('@BINDERS@', '\n'.join('    ' + b for b in sb + pb))
    open(f'{d}/Part_{part}_KeySource.lean', 'w').write(body)
    break


def fresh_links(d, part, ab):
    """`Part_<c_j>_KeyFresh.lean`: the keys of the six fresh columns (a core theorem up to the
    sampled members, then one theorem per column)."""
    import re
    from patterns import parse
    on_ns, pre_ns = f'Stage_online_eval_{part}', f'Stage_preprocess_{part}'
    _, pnames, patoms = parse(f'{d}/generated/{pre_ns}.lean', 'generatedRoot')
    PA = dict(zip(pnames, patoms))
    on_cols = [(394, 413), (419, 438), (441, 460), (463, 482), (485, 504), (507, 526)]
    pre_cols = [(220, 233), (239, 252), (255, 268), (271, 284), (287, 300), (303, 316)]
    on_atoms = ['h386_5', 'h387_5', 'h390_5', 'h391_5', 'h392_5', 'h393_5']
    pre_atoms = ['h210_7', 'h211_3', 'h214_7', 'h215_3', 'h217_3', 'h219_3']
    ob = chain_binders(f'{d}/generated/{on_ns}.lean', on_ns, [], ab, on_atoms)
    pb = [prefixed(b) for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab, pre_atoms)]
    cell = lambda h, ph, o, p: f"""  have hc{h} : ∀ i : Fin 240, w_{h}_0 i = pw_{ph}_0 i ∧ w_{h}_1 i = pw_{ph}_1 i ∧
      w_{h}_2 i = pw_{ph}_2 i ∧ w_{h}_3 i = pw_{ph}_3 i := fun i ↦
    keysCell{h} (hon := h{h}_5 i) (hpre := ph{ph}_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm{h} : w_{o}_0 = pw_{p}_0 ∧ w_{o}_1 = pw_{p}_1 ∧ w_{o}_2 = pw_{p}_2 ∧ w_{o}_3 = pw_{p}_3 := by
    have a := fun o ↦ members{o} (h{o}_5 o)
    have b := fun o ↦ premembers{p} (ph{p}_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).2.2.2
"""
    body = f"""import Part_{part}_KeysLuts
import Part_{part}_KeysCells
import Part_{part}_Sums
import Part_{part}_PreSums
import Part_{part}_PrePlace
import Part_{part}_PreRootSums
import Part_{part}_RootSums
import Part_{part}_FreshCarrier

/-! The fresh columns' keys of the online root of part `{part}` are those the preprocessing computes
(written by `generator/gen_key_links.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The sampled members of the fresh PRF: the online root's keys are the preprocessing's. -/
theorem freshKeyCore {{model : HashModel}} {{params : {on_ns}.Params}} {{params' : {pre_ns}.Params}}
""" + '\n'.join('    ' + b for b in ob + pb) + f"""
    (hk : w_363_0 = pw_201_0)
    (ho0 : w_377_0 = pw_202_0) (ho1 : w_378_0 = pw_203_0) (ho2 : w_381_0 = pw_204_0) (ho3 : w_382_0 = pw_205_0)
    (hf0 : w_365_0 = pw_216_0) (hf1 : w_366_0 = pw_216_1) (hf2 : w_367_0 = pw_216_2) (hf3 : w_368_0 = pw_216_3)
    (hg0 : w_358_0 = pw_218_0) (hg1 : w_359_0 = pw_218_1) (hg2 : w_360_0 = pw_218_2) (hg3 : w_361_0 = pw_218_3)
    (hc0 : w_384_0 = pw_210_6) (hc1 : w_385_0 = pw_210_7) (hc2 : w_388_0 = pw_214_6) (hc3 : w_389_0 = pw_214_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_372_0, w_375_0, w_373_0, w_376_0] c k =
      model.sample Q7 N 1 14 w_363_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hg0 hg1 hg2 hg3 hc0 hc1 hc2 hc3
{cell(386, 210, 387, 211)}  obtain ⟨m0, m1, m2, m3⟩ := hm386
  rw [m0, m1, m2, m3] at h390_5
{cell(390, 214, 391, 215)}  obtain ⟨n0, n1, n2, n3⟩ := hm390
  rw [n0, n1, n2, n3] at h392_5
  have hr : ∀ o : Fin 48, w_392_0 o = pw_217_0 o ∧ w_392_1 o = pw_217_1 o ∧ w_392_2 o = pw_217_2 o ∧
      w_392_3 o = pw_217_3 o := by
    intro o
    have a := keyRound392 (h392_5 o)
    have b := preKeyRound217 (ph217_3 o)
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  intro o
  have a := keyRound393 (h393_5 o)
  have b := preKeyRound219 (ph219_3 o)
  obtain ⟨r0, r1, r2, r3⟩ := hr o
  rw [r0, r1, r2, r3, ← b] at a
  simp only [Prod.mk.injEq] at a
  exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
"""
    for k, ((P, S), (PP, PS)) in enumerate(zip(on_cols, pre_cols)):
        ofams = [x for i in range(6) for x in (f'h{P + 1 + 3 * i}_0', f'h{P + 2 + 3 * i}_0')]
        pfams = [n for n in pnames if re.match(rf'MxxRuntime\.familyGetStatic w_{PP}_\d \(\d\) ', PA[n])]
        oinit = ' '.join(f'e_{P + 3 + 3 * i}_0' for i in range(6))
        pa = PA[f'h{PS}_3']
        pinit = ' '.join('pe_' + w[2:] for w in re.findall(r'w_\d+_0', re.search(
            r'\(Int\.toNat \(\d+\)\) \((.*?)\) \(w_' + str(PS), pa).group(1)))
        cb = chain_binders(f'{d}/generated/{on_ns}.lean', on_ns, [], ab, [f'h{P}_5'] + ofams + [f'h{S}_5'])
        cpb = [prefixed(b) for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab,
                                                  [f'h{PP}_3'] + pfams + [f'h{PS}_3'])]
        shift = f' + {8 * k}' if k else ''
        body += f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `{k}`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol{k} {{model : HashModel}} {{params : {on_ns}.Params}} {{params' : {pre_ns}.Params}}
""" + '\n'.join('    ' + b for b in cb + cpb) + f"""
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_{S}_1 = pw_{PS}_1 ∧ w_{S}_3 = pw_{PS}_3 := by
  have hp : ∀ o : Fin 8, w_{P}_0 o = pw_{PP}_0 o ∧ w_{P}_1 o = pw_{PP}_1 o ∧ w_{P}_2 o = pw_{PP}_2 o ∧
      w_{P}_3 o = pw_{PP}_3 o := by
    intro o
    have a := place{P} (h := h{P}_5 o)
      (x := (w_393_0 ⟨o.val{shift}, by omega⟩, w_393_1 ⟨o.val{shift}, by omega⟩,
        w_393_2 ⟨o.val{shift}, by omega⟩, w_393_3 ⟨o.val{shift}, by omega⟩,
        w_393_4 ⟨o.val{shift}, by omega⟩, w_393_5 ⟨o.val{shift}, by omega⟩, ()))
    have b := prePlace{PP} (h := ph{PP}_3 o)
      (x := (pw_219_0 ⟨o.val{shift}, by omega⟩, pw_219_1 ⟨o.val{shift}, by omega⟩,
        pw_219_2 ⟨o.val{shift}, by omega⟩, pw_219_3 ⟨o.val{shift}, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val{shift}, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst {oinit} {pinit}
  have a := rootSum{S} {' '.join(ofams)} h{S}_5
  have b := prerootSum{PS} {' '.join('p' + x for x in pfams)} ph{PS}_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩
"""
    body += f'\nend RefreshTopProof.Part_{part}\n'
    open(f'{d}/Part_{part}_KeyFresh.lean', 'w').write(body)


def tuple_wires(atom):
    """The input tuple items of a parallel atom (`w i` gives `w`)."""
    import re
    from chain import WIRE
    i = atom.index('params i (') + len('params i (')
    depth, j = 1, i
    while depth:
        depth += {'(': 1, ')': -1}.get(atom[j], 0)
        j += 1
    return WIRE.findall(atom[i:j])


def smudge_links(d, part, ab):
    """`Part_<c_j>_KeySmudge.lean`: the smudge keys of every decoder."""
    import re
    from patterns import parse
    on_ns, pre_ns = f'Stage_online_eval_{part}', f'Stage_preprocess_{part}'
    _, onames, oatoms = parse(f'{d}/generated/{on_ns}.lean', 'generatedRoot')
    _, pnames, patoms = parse(f'{d}/generated/{pre_ns}.lean', 'generatedRoot')
    OA, PA = dict(zip(onames, oatoms)), dict(zip(pnames, patoms))
    total = open(f'{d}/Part_{part}_SmudgeTotal.lean').read()
    luts = [int(x) for x in re.findall(r'\(hlut : ∀ o : Fin 160, \S+\.parallel_generatedRoot_(\d+) ', total)]
    cells = open(f'{d}/Part_{part}_KeysCells.lean').read()
    body = f"""import Part_{part}_KeysLuts
import Part_{part}_KeysCells
import Part_{part}_Sums
import Part_{part}_PreSums
import Part_{part}_PreRootSums
import Part_{part}_RootSums

/-! The smudge keys of the online root of part `{part}` are those the preprocessing computes
(written by `generator/gen_key_links.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof
"""
    for dec, L in enumerate(luts):
        C, S = L - 6, L + 19
        pc = int(re.search(rf'theorem keysCell{C} [\s\S]*?parallel_generatedRoot_(\d+) Backend\.backend model params\' i', cells).group(1))
        pc2 = int(re.search(rf'theorem keysCell{C + 4} [\s\S]*?parallel_generatedRoot_(\d+) Backend\.backend model params\' i', cells).group(1))
        # The preprocessing's members, LUT and sum after its cells.
        pm = next(int(re.search(r'parallel_generatedRoot_(\d+)', PA[n]).group(1)) for n in pnames
                  if re.match(rf'\(∀ i : Fin 160, parallel_generatedRoot_\d+ backend hashModel params i \(w_{pc}_0, ', PA[n]))
        pm2 = next(int(re.search(r'parallel_generatedRoot_(\d+)', PA[n]).group(1)) for n in pnames
                   if re.match(rf'\(∀ i : Fin 160, parallel_generatedRoot_\d+ backend hashModel params i \(w_{pc2}_0, ', PA[n]))
        pl = next(int(re.search(r'parallel_generatedRoot_(\d+)', PA[n]).group(1)) for n in pnames
                  if re.search(rf'parallel_generatedRoot_\d+ backend hashModel params i \(w_\d+_0, \(\(w_{pm2}_0 i\)', PA[n]))
        ps = next(n for n in pnames if PA[n].startswith('MxxIR.IterRuns') and f'w_{pl}_0, (w_{pl}_1' in PA[n])
        PS = int(ps[1:].split('_')[0])
        oc, ol = tuple_wires(OA[f'h{C}_5']), tuple_wires(OA[f'h{L}_5'])
        pct, pct2, plt = tuple_wires(PA[f'h{pc}_7']), tuple_wires(PA[f'h{pc2}_7']), tuple_wires(PA[f'h{pl}_3'])
        ofams = [n for n in onames if re.match(rf'MxxRuntime\.familyGetStatic w_{L}_\d \(\d\) ', OA[n])]
        pfams = [n for n in pnames if re.match(rf'MxxRuntime\.familyGetStatic w_{pl}_\d \(\d\) ', PA[n])]
        oa = OA[f'h{S}_5']
        oinit = ' '.join('e_' + w[2:] for w in re.findall(r'w_\d+_0', re.search(rf'\(Int\.toNat \(\d+\)\) \((.*?)\) \(w_{S}', oa).group(1)))
        pa = PA[ps]
        pinit = ' '.join('pe_' + w[2:] for w in re.findall(r'w_\d+_0', re.search(rf'\(Int\.toNat \(\d+\)\) \((.*?)\) \(w_{PS}', pa).group(1)))
        on_atoms = [f'h{C}_5', f'h{C + 1}_5', f'h{C + 4}_5', f'h{C + 5}_5', f'h{L}_5'] + ofams + [f'h{S}_5']
        pre_atoms = [f'h{pc}_7', f'h{pm}_3', f'h{pc2}_7', f'h{pm2}_3', f'h{pl}_3'] + pfams + [ps]
        ob = chain_binders(f'{d}/generated/{on_ns}.lean', on_ns, [], ab, on_atoms)
        pb = [prefixed(b) for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab, pre_atoms)]
        links = f"""    (hk : {oc[13]} = p{pct[0]})
    (ho0 : {oc[6]} = p{pct[1]}) (ho1 : {oc[7]} = p{pct[2]}) (ho2 : {oc[10]} = p{pct[3]}) (ho3 : {oc[11]} = p{pct[4]})
    (hf0 : {ol[0]} = p{plt[3]}) (hf1 : {ol[1]} = p{plt[7]}) (hf2 : {ol[2]} = p{plt[6]}) (hf3 : {ol[3]} = p{plt[8]})
    (hc0 : {oc[14]} = pw_{pc}_6) (hc1 : {oc[15]} = pw_{pc}_7)
    (hc2 : {tuple_wires(OA[f'h{C + 4}_5'])[14]} = pw_{pc2}_6) (hc3 : {tuple_wires(OA[f'h{C + 4}_5'])[15]} = pw_{pc2}_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![{oc[1]}, {oc[4]}, {oc[2]}, {oc[5]}] c k =
      model.sample Q6 N 1 12 {oc[13]} (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))"""
        mem = lambda h, ph, o, p: f"""  have hc{h} : ∀ i : Fin 800, w_{h}_0 i = pw_{ph}_0 i ∧ w_{h}_1 i = pw_{ph}_1 i ∧
      w_{h}_2 i = pw_{ph}_2 i ∧ w_{h}_3 i = pw_{ph}_3 i := fun i ↦
    keysCell{h} (hon := h{h}_5 i) (hpre := ph{ph}_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm{h} : w_{o}_0 = pw_{p}_0 ∧ w_{o}_1 = pw_{p}_1 ∧ w_{o}_2 = pw_{p}_2 ∧ w_{o}_3 = pw_{p}_3 := by
    have a := fun o ↦ members{o} (h{o}_5 o)
    have b := fun o ↦ premembers{p} (ph{p}_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc{h} _).2.2.2
"""
        body += f"""
set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `{dec}`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink{dec} {{model : HashModel}} {{params : {on_ns}.Params}} {{params' : {pre_ns}.Params}}
""" + '\n'.join('    ' + b for b in ob + pb) + '\n' + links + f""" :
    w_{S}_1 = pw_{PS}_1 ∧ w_{S}_3 = pw_{PS}_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
{mem(C, pc, C + 1, pm)}  obtain ⟨m0, m1, m2, m3⟩ := hm{C}
  rw [m0, m1, m2, m3] at h{C + 4}_5
{mem(C + 4, pc2, C + 5, pm2)}  obtain ⟨n0, n1, n2, n3⟩ := hm{C + 4}
  rw [n0, n1, n2, n3] at h{L}_5
  have hl : ∀ o : Fin 160, w_{L}_0 o = pw_{pl}_0 o ∧ w_{L}_1 o = pw_{pl}_1 o ∧ w_{L}_2 o = pw_{pl}_2 o ∧
      w_{L}_3 o = pw_{pl}_3 o := by
    intro o
    have a := keyRound{L} (h{L}_5 o)
    have b := preKeyRound{pl} (ph{pl}_3 o)
{'' if L == luts[0] else f'    rw [show smudgeWeight{L} = smudgeWeight{luts[0]} from rfl] at a' + chr(10)}    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst {oinit} {pinit}
  have a := rootSum{S} {' '.join(ofams)} h{S}_5
  have b := prerootSum{PS} {' '.join('p' + x for x in pfams)} p{ps}
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩
"""
    body += f'\nend RefreshTopProof.Part_{part}\n'
    open(f'{d}/Part_{part}_KeySmudge.lean', 'w').write(body)


for part in parts_of(d):
    fresh_links(d, part, ab)
    smudge_links(d, part, ab)
    break
