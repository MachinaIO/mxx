"""Write `TopTables.lean`: the scalar constants of the instances of the merged stages.

Instance `i0` of a merged stage (part `(i0 / 2, i0 % 2)`) reads its constants from tables
`sel_k (i0 : Int)` of the stage files. Each constant of `CONSTANTS` is the natural number of its
table (an abbreviation of the exported table of the first stage having it), with a `simp` lemma
`((c i0 : Nat) : Int) = Stage_<phase>.sel_k (i0 : Int)` for every stage table with its values,
so statements may use `c i0` where the stages compute with the table.

Usage: `python3 gen_tables.py <parameter directory>`.
"""
import re
import sys

q = [1073741441, 1073741329, 1073740609, 1073740529]
# (name, token, description, values) — the tables of chain `c = i0 / 2`, component `j = i0 % 2`.
CONSTANTS = [
    ('qd', 'Qdi0', 'The prime of `Q₀` that the decoding ring `R` drops (`Q₀ = qd R`).',
     [q[3]] * 6 + [q[2]] * 2),
    ('rp', 'Rpi0', 'The third prime of the decoding ring (`R = q₀ q₁ rp`).', [q[2]] * 6 + [q[3]] * 2),
    ('pp', 'Ppi0', 'The prime `p` of `R = p Mx`.', [q[2]] * 4 + [q[1]] * 4),
    ('mxp', 'Mxpi0', 'The second prime of the plaintext modulus (`Mx = q₀ mxp`).',
     [q[1]] * 4 + [q[2]] * 2 + [q[3]] * 2),
    ('qcN', 'QcNi0', 'The CRT prime `q_c` of the chain.', [q[0]] * 2 + [q[1]] * 2 + [q[2]] * 2 + [q[3]] * 2),
    ('comp', 'Compi0', 'The output component `j`.', [0, 1] * 4),
]


def _chain_table(key):
    """The table of the chain constant `key` of `chains.CHAINS` over the instances."""
    from chains import CHAINS
    names = {'q0': q[0], 'q1': q[1], 'q2': q[2], 'q3': q[3]}
    return [int(names.get(str(CHAINS[i // 2][key]), CHAINS[i // 2][key])) for i in range(8)]


# The constants of the decoding chain (`chains.py`): `R = Qq qc`, `p cp = Qq k` with
# `k = cp / kdiv`, `k k3 k1 ≡ q3 (mod qc)`, the mask table `(0, h, 0, h)`, `half = ⌈R / 2⌉`.
CONSTANTS += [
    ('Qq', 'Qqi0', 'The quotient `Qq = R / qc`.', _chain_table('Qq')),
    ('cp', 'Cpi0', 'The action constant `c\'` (`p c\' = Qq k`).', _chain_table('cp')),
    ('kdiv', 'Kdivi0', 'The divisor of the action constant (`k = c\' / kdiv`).', _chain_table('kdiv')),
    ('k3', 'K3i0', 'The first online scale factor.', _chain_table('k3')),
    ('k1', 'K1i0', 'The second online scale factor.', _chain_table('k1')),
    ('hm', 'Hmi0', 'The mask sampling value `h` (`2 h = qc + 1`).', _chain_table('h')),
    ('half', 'Halfi0', 'The half of the decoding ring.', _chain_table('half')),
    ('inv4', 'Inv4i0', 'The inverse `(3 R + 1) / 4` of `4` modulo the decoding ring.',
     [(3 * (q[0] * q[1] * p) + 1) // 4 for p in [q[2]] * 6 + [q[3]] * 2]),
]


def main(d):
    stages = {}
    for phase in ('preprocess', 'encrypt', 'online_eval'):
        text = open(f'{d}/generated/Stage_{phase}.lean').read()
        stages[phase] = {k: [int(x) for x in v.split(',')] for k, v in re.findall(
            r'^abbrev (sel_\d+) \(i : Int\) : Int := MxxIR\.selectInt i \[([^\]]*)\]', text, re.M)}
    out = ['''import Stage_preprocess
import Stage_encrypt
import Stage_online_eval
import Mathlib.Algebra.Ring.Int.Defs

/-! The scalar constants of the instances of the merged stages (written by
`generator/gen_tables.py`). -/

namespace RefreshTopProof

theorem selectInt_nonneg {i : Int} {l : List Int} (h : ∀ x ∈ l, 0 ≤ x) :
    0 ≤ MxxIR.selectInt i l := by
  unfold MxxIR.selectInt
  cases hx : l[i.toNat]? with
  | none => simp
  | some x => exact h x (List.mem_of_getElem? hx)

''']
    for name, _, doc, values in CONSTANTS:
        tables = [(p, k) for p, ts in stages.items() for k, v in ts.items() if v == values]
        listed = ', '.join(map(str, values))
        out.append(f'''/-- {doc} (the table, over the instance index) -/
def {name}Z (i : Int) : Int := MxxIR.selectInt i [{listed}]

''')
        for p, k in tables:
            out.append(f'@[simp] theorem {name}Z_{p}_{k} (i : Int) : Stage_{p}.{k} i = {name}Z i := rfl\n')
        out.append(f'''
/-- {doc} -/
def {name} (i0 : Nat) : Nat := ({name}Z (i0 : Int)).toNat

@[simp] theorem {name}_cast (i0 : Nat) : (({name} i0 : Nat) : Int) = {name}Z (i0 : Int) :=
  Int.toNat_of_nonneg (selectInt_nonneg (by decide))

@[simp] theorem {name}_cast_ring {{S : Type*}} [Ring S] (i0 : Nat) :
    (({name} i0 : Nat) : S) = (({name}Z (i0 : Int) : Int) : S) := by
  rw [← {name}_cast, Int.cast_natCast]

''')
    # The instance moduli: the stages' modulus abbreviations (`mod_k`), so types unify by unfolding.
    text = open(f'{d}/generated/Stage_online_eval.lean').read()
    mods = dict((body, k) for k, body in re.findall(
        r'^abbrev (mod_\d+) \(i : Int\) : Nat := \((.*)\)\.toNat$', text, re.M))

    def online(values):
        return next(k for k, v in stages['online_eval'].items() if v == values)

    def modulus(form):
        return mods[form]
    rp_ = online([q[2]] * 6 + [q[3]] * 2)
    qc_ = online([q[0]] * 2 + [q[1]] * 2 + [q[2]] * 2 + [q[3]] * 2)
    mx_ = online([q[1]] * 4 + [q[2]] * 2 + [q[3]] * 2)
    r_mod = modulus(f'(1073741441) * (1073741329) * (({rp_} (i)))')
    qc_mod = modulus(f'(({qc_} (i)))')
    mx_mod = modulus(f'(1073741441) * (({mx_} (i)))')
    out.append(f'''/-- The decoding ring of instance `i0`: `q₀ q₁ q₂` (chains `0`–`2`) or `q₀ q₁ q₃` (chain `3`). -/
abbrev R (i0 : Nat) : Nat := Stage_online_eval.{r_mod} (i0 : Int)

/-- The CRT prime `q_c` of the chain of instance `i0`, the modulus of its residues. -/
abbrev qc (i0 : Nat) : Nat := Stage_online_eval.{qc_mod} (i0 : Int)

/-- The plaintext modulus of instance `i0`: `q₀ q₁` (chains `0`, `1`), `q₀ q₂`, `q₀ q₃`. -/
abbrev Mx (i0 : Nat) : Nat := Stage_online_eval.{mx_mod} (i0 : Int)

''')
    casts = ', '.join(f'{name}_cast' for name, _, _, _ in CONSTANTS)
    out.append(f'''/-- `exact h` up to the casts of the instance's constants (natural numbers to their tables). -/
macro "cast_exact " h:term : tactic => `(tactic| first
  | exact $h
  | (have h' := $h; simp only [{casts}] at h' ⊢; exact h'))

''')
    out.append('end RefreshTopProof\n')
    open(f'{d}/TopTables.lean', 'w').write(''.join(out))


if __name__ == '__main__':
    main(sys.argv[1])
