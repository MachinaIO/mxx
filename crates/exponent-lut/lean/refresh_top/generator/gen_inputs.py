"""Write `Part_<c_j>_Inputs.lean`: the semantic facts the online chains assume about the online
root's inputs, proven from the preprocessing's (prefix `p`) and the encryption's (prefix `q`)
atoms, about the wires the inputs come from.

Usage: `python3 gen_inputs.py <parameter directory>`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from chain import chain_binders, prefixed, seg_args
from gen_part import parts_of
from chains import chain_of
from segment import abbreviations

def smudge_inputs(d, part):
    import re
    from patterns import parse
    from stage_map import input_sources, outputs_of
    from gen_key_links import tuple_wires
    on_ns, pre_ns, enc_ns = f'Stage_online_eval_{part}', f'Stage_preprocess_{part}', f'Stage_encrypt_{part}'
    _, on, oa = parse(f'{d}/generated/{on_ns}.lean', 'generatedRoot')
    _, pn, pa = parse(f'{d}/generated/{pre_ns}.lean', 'generatedRoot')
    _, en, ea = parse(f'{d}/generated/{enc_ns}.lean', 'generatedRoot')
    OA, PA, EA = dict(zip(on, oa)), dict(zip(pn, pa)), dict(zip(en, ea))
    src = dict(input_sources(d, 'online_eval', part))
    pout, eout = outputs_of(d, 'preprocess', part), outputs_of(d, 'encrypt', part)
    def source(w):
        st, name, k = src[w]
        return (pout if 'preprocess' in name else eout)[k]
    def producer(A, names, w):
        return next(n for n in names if n != '-' and re.search(rf'\b{w}$', A[n]))
    total = open(f'{d}/Part_{part}_SmudgeTotal.lean').read()
    luts = [int(x) for x in re.findall(r'\(hlut : ∀ o : Fin 160, \S+\.parallel_generatedRoot_(\d+) ', total)]
    cells = open(f'{d}/Part_{part}_KeysCells.lean').read()
    er = open(f'{d}/Part_{part}_EncRoot.lean').read()
    tpl = open(os.path.join(os.path.dirname(__file__), 'templates', 'InputsSmudge.thm')).read()
    out = ''
    for k, L in enumerate(luts):
        C = L - 6
        c1 = int(re.search(rf'theorem keysCell{C} [\s\S]*?parallel_generatedRoot_(\d+) Backend\.backend model params\' i', cells).group(1))
        c2 = int(re.search(rf'theorem keysCell{C + 4} [\s\S]*?parallel_generatedRoot_(\d+) Backend\.backend model params\' i', cells).group(1))
        oc, ol = tuple_wires(OA[f'h{C}_5']), tuple_wires(OA[f'h{L}_5'])
        S = int(source(ol[0]).split('_')[1])
        H = int(source(ol[4]).split('_')[1])
        Lb = int(source(oc[1]).split('_')[1])
        one00 = source(oc[6])
        one = next(m.group(1) for m in re.finditer(r'theorem (encOne_\d+_0) ', er)
                   if re.search(r'Rel sm !\[!\[' + one00 + ',', er[m.start():er.index(':= by', m.start())]))
        oh = er[er.index(f'theorem {one} '):]
        oh = oh[:oh.index(':= by')]
        keys = re.search(r'Rel sm !\[!\[(w_\d+_0), (w_\d+_0)\], !\[(w_\d+_0), (w_\d+_0)\]\]', oh).groups()
        rows = re.search(r'!\[(w_\d+_0), (w_\d+_0)\] 26', oh).groups()
        mc, me = re.search(r'message \(n := N\) (w_\d+_0) (w_\d+_0)', oh).groups()
        pc1, pc2 = tuple_wires(PA[f'h{c1}_7']), tuple_wires(PA[f'h{c2}_7'])
        eh, elab = tuple_wires(EA[f'h{H}_1']), tuple_wires(EA[f'h{Lb}_6'])
        psec, pone = pc1[5], pc1[8]
        qsm, qone = eh[0], eh[2]
        rep = {'@K@': str(k), '@C1@': str(c1), '@C2@': str(c2), '@S@': str(S), '@H@': str(H), '@L@': str(Lb),
               '@ONE@': one, '@QMC@': 'q' + mc, '@QME@': 'q' + me,
               '@O00@': 'q' + keys[0], '@O01@': 'q' + keys[1], '@O10@': 'q' + keys[2], '@O11@': 'q' + keys[3],
               '@OC0@': 'q' + rows[0], '@OC1@': 'q' + rows[1], '@QKEY@': 'q' + eh[1], '@PKEY@': 'p' + pc1[0],
               '@PC@': 'p' + pc1[6], '@PE@': 'p' + pc1[7], '@PC2@': 'p' + pc2[6], '@PE2@': 'p' + pc2[7],
               '@PSEC@': 'p' + psec, '@PHSEC@': 'p' + producer(PA, pn, psec),
               '@PONE@': 'p' + pone, '@PHONE@': 'p' + producer(PA, pn, pone),
               '@QSM@': 'q' + qsm, '@QHSM@': 'q' + producer(EA, en, qsm),
               '@QONE@': 'q' + qone, '@QHONE@': 'q' + producer(EA, en, qone),
               '@QLC@': 'q' + elab[6], '@QLE@': 'q' + elab[7],
               '@PRE@': pre_ns, '@ENC@': enc_ns}
        pb = [prefixed(b, 'p') for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab,
            [producer(PA, pn, psec), producer(PA, pn, pone), f'h{c1}_7', f'h{c2}_7', f'h{S}_3'])]
        eb = [prefixed(b, 'q').replace("params'", "params''") for b in chain_binders(
            f'{d}/generated/{enc_ns}.lean', enc_ns, [(er, one)], ab,
            [f'h{H}_1', f'h{Lb}_6', producer(EA, en, qsm), producer(EA, en, qone)])]
        t = tpl
        for a, b in rep.items():
            t = t.replace(a, b)
        t = t.replace('@BINDERS@', '\n'.join('    ' + b for b in pb + eb))
        t = t.replace('@ARGS@', ' '.join(f'({a} := q{a})' for a in seg_args(er, one)))
        out += t
    return out


d = sys.argv[1]
ab = abbreviations(open(f'{d}/TopSetup.lean').read())
TPL = open(os.path.join(os.path.dirname(__file__), 'templates', 'Inputs.body')).read()
for part in parts_of(d):
    pre_ns, enc_ns = f'Stage_preprocess_{part}', f'Stage_encrypt_{part}'
    er = open(f'{d}/Part_{part}_EncRoot.lean').read()
    pre_atoms = ['h31_0', 'h34_0', 'h47_0', 'h52_0', 'h53_0', 'h54_0', 'h62_0', 'h66_0', 'h81_0', 'h85_0',
                 'h91_0', 'h102_0', 'h103_0', 'h104_0', 'h112_0', 'h77_0', 'h126_0', 'h116_0', 'h130_0',
                 'h136_0', 'h35_7', 'h39_7', 'h41_3', 'h43_3']
    pb = [prefixed(b, 'p') for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab, pre_atoms,
                                                  ['w_67_0', 'w_92_0', 'w_117_0', 'w_137_0'])]
    eb = [prefixed(b, 'q').replace("params'", "params''") for b in chain_binders(
        f'{d}/generated/{enc_ns}.lean', enc_ns, [(er, 'encOne_359_0')], ab, ['h375_6', 'h379_1', 'h380_1'])]
    body = TPL.replace('@MASKH@', str(chain_of(part)['h'])).replace('@P@', part).replace('@PRE@', pre_ns).replace('@ENC@', enc_ns)
    body = body.replace('@BINDERS@', '\n'.join('    ' + b for b in pb + eb))
    body = body.replace('@ARGS(encOne_359_0)@', ' '.join(f'({a} := q{a})' for a in seg_args(er, 'encOne_359_0')))
    # The fresh chain.
    fpb = [prefixed(b, 'p') for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab,
                                                   ['h206_0', 'h209_0', 'h210_7', 'h214_7', 'h216_3', 'h218_3'])]
    feb = [prefixed(b, 'q').replace("params'", "params''") for b in chain_binders(
        f'{d}/generated/{enc_ns}.lean', enc_ns, [(er, 'encOne_315_0')], ab, ['h331_6', 'h335_1', 'h336_1'])]
    fresh = open(os.path.join(os.path.dirname(__file__), 'templates', 'InputsFresh.thm')).read()
    fresh = fresh.replace('@PRE@', pre_ns).replace('@ENC@', enc_ns)
    fresh = fresh.replace('@BINDERS@', '\n'.join('    ' + b for b in fpb + feb))
    fresh = fresh.replace('@ARGS(encOne_315_0)@', ' '.join(f'({a} := q{a})' for a in seg_args(er, 'encOne_315_0')))
    body = body.replace(f'end RefreshTopProof.Part_{part}', fresh + f'end RefreshTopProof.Part_{part}')
    body = body.replace(f'end RefreshTopProof.Part_{part}', smudge_inputs(d, part) + f'end RefreshTopProof.Part_{part}')
    # The source LUT.
    spb = [prefixed(b, 'p') for b in chain_binders(f'{d}/generated/{pre_ns}.lean', pre_ns, [], ab, ['h182_3'])]
    seb = [prefixed(b, 'q').replace("params'", "params''") for b in chain_binders(
        f'{d}/generated/{enc_ns}.lean', enc_ns, [(er, 'encOne_280_0')], ab, ['h291_1'], ['w_7_0'])]
    src = open(os.path.join(os.path.dirname(__file__), 'templates', 'InputsSource.thm')).read()
    src = src.replace('@PRE@', pre_ns).replace('@ENC@', enc_ns).replace('@MX@', chain_of(part)['Mx']).replace('@PS@', chain_of(part)['p'])
    src = src.replace('@BINDERS@', '\n'.join('    ' + b for b in spb + seb))
    src = src.replace('@ARGS@', ' '.join(f'({a} := q{a})' for a in seg_args(er, 'encOne_280_0')))
    body = body.replace(f'end RefreshTopProof.Part_{part}', src + f'end RefreshTopProof.Part_{part}')
    open(f'{d}/Part_{part}_Inputs.lean', 'w').write(body)
    break
