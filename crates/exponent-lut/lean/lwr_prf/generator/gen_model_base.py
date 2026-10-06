#!/usr/bin/env python3
"""LwrModelBase.lean: the inputs of the model of a run.

Usage: gen_model_base.py <parameter dir>

`LwrModelBase` of `average-n32-crt7x40` is the template: the hash-derived switching tops of the
readout LUT, the clamped secrets, and the plaintexts and constants of the model, with the counts,
the moduli and the stride of the parameter set.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from info import Info, wrap  # noqa: E402
from lean_patterns import proj, destruct  # noqa: E402
from gen_encrypt import split_top  # noqa: E402

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'average-n32-crt7x40')


def lutb_prefix(I):
    for m in re.finditer(r'hashSample \(hashModel\) \(\[([0-9, ]*)\]\)', I.pre):
        text = bytes(int(x) for x in m.group(1).split(', ')).decode()
        if '/lut-b/' in text:
            return m.group(1), text
    raise AssertionError('no lut-b tag')


def module(I):
    L, B, C, M, W, s = I.L, I.B, I.C, I.M, I.W, I.stride
    m1 = I.m(1)
    t = open(os.path.join(TEMPLATES, 'LwrModelBase.lean')).read()
    raw, text = lutb_prefix(I)
    # The switching tops: the proof destructures the LUT cell of the preprocessing root.
    path = f'{I.d}/generated/Stage_preprocess.lean'
    a = t.index('set_option maxHeartbeats 8000000 in\n/-- The switching tops')
    b = t.index('/-! ## Inputs -/')
    ins = ', '.join(['key', 'payload'] + [f'b{j}, x{j}' for j in range(B)])
    binders = ' '.join(f'b{j} x{j}' for j in range(B))
    haves = proj(path, 'generatedRoot', ['h3_3', 'hout'])
    lines = destruct(path, 'parallel_generatedRoot_3', 'hcell', ['h1_0', 'h2_0', 'hout'])
    top = f'''set_option maxHeartbeats 8000000 in
/-- The switching tops of the readout LUT are hash-derived. -/
theorem preprocess_switchTop {{model : HashModel}} {{key : ByteArray}} {{payload : ExactMatrix Q{L} N 1 1}}
    {{{binders} : Fin {C} → Int}} {{outputs}}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      ({ins}, ()) outputs) (i : Fin {W}) :
    ![outputs.1 i, outputs.2.1 i] = lutB model key i := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
{haves}  subst hout
  have hcell := h3_3 i
  unfold Stage_preprocess.parallel_generatedRoot_3 at hcell
{lines}  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, -⟩ := hout
  funext j
  fin_cases j
  · simp only []
    rw [ho0, hashSample_eq h1_0]
    rfl
  · simp only []
    rw [ho1, hashSample_eq h2_0]
    rfl

'''
    t = t[:a] + top + t[b:]
    old_prefix = t[t.index('/-- `rns-bgg/lut-b/'):t.index('/-- The hash-derived switching tops')]
    t = t.replace(old_prefix, f'''/-- `{text}` -/
def lutBPrefix : Blob :=
  [{wrap(raw.split(', '), ', ', 4, 3)}]

''')
    sel = wrap([f'ext.input_{6 + 2 * b}' for b in range(B)], ', ', 4, 40)
    reps = [
        ('def selOf : Fin 5 → Fin 14 → Int := ![ext.input_6, ext.input_8, ext.input_10, ext.input_12, ext.input_14]',
         f'def selOf : Fin {B} → Fin {C} → Int :=\n  ![{sel}]'),
        ('Fin 16', f'Fin {W}'), ('ExactMatrix Q1 N 1 2', f'ExactMatrix Q1 N 1 {m1}'),
        ('model.sample Q1 N 1 2', f'model.sample Q1 N 1 {m1}'),
        ('ExactMatrix Q7 N 1 1', f'ExactMatrix Q{L} N 1 1'), ('reducePoly Q7 N', f'reducePoly Q{L} N'),
        ('Fin 17', f'Fin {M}'), ('Fin 14', f'Fin {C}'), ('Fin 5', f'Fin {B}'),
        ('X^(2 a_k)', f'X^({s} a_k)'), ('Xe (2 * (ext.input_3 k).toNat)', f'Xe ({s} * (ext.input_3 k).toNat)'),
        ('X^(2 z_b)', f'X^({s} z_b)'), ('Xe (2 * zOf ext b)', f'Xe ({s} * zOf ext b)'),
    ]
    for x, y in reps:
        assert x in t, x
        t = t.replace(x, y)
    t = t.replace('# The model of a run (prototype)', '# The model of a run')
    t = t.replace('\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n',
                  '\nopen Mxx.Primitives MxxRuntime MxxExponentLut\n\nset_option maxRecDepth 100000\n', 1)
    left = re.findall(r'\bQ7\b|Fin 17\b|Fin 14\b|Fin 16\b', t)
    assert (L, M, C, W) == (7, 17, 14, 16) or not left, left[:5]
    return t


def main():
    d = sys.argv[1]
    I = Info(d)
    path = os.path.join(d, 'LwrModelBase.lean')
    text = module(I)
    with open(path, 'w') as f:
        f.write(text)
    print('wrote', path)


if __name__ == '__main__':
    main()
