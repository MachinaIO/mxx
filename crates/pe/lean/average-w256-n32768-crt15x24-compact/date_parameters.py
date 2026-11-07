"""Parameter specialization of the reference proof's Lean templates."""

import re
from pathlib import Path

from export_format import translate

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "average-w128-n65536-crt22x32-compact"
WIDTH = 256
STEPS = 2
DIGITS = 30
N = 32768
BASE = 4096
CUTOFF = 2048
HELPERS = 45
SWITCHES = 30
LEAVES = 225
STRIDE = 256
KEY_CUTOFF = 2610114297


def numeric_aliases(module):
    return list(map(int, re.findall(r'local notation "mxxNat\d+" => (\d+)',
                                  (HERE / "generated" / f"{module}.lean").read_text())))


Q, PQ, P = numeric_aliases("Stage_encrypt")
KEY_NUMERATOR, KEY_DENOMINATOR = numeric_aliases("Stage_keygen")[2:]
OLD_SWITCHES = [1 + 1024 * index for index in [*range(1, 16), *range(16, 128, 16)]]
NEW_SWITCHES = [1 + STRIDE * index for index in [*range(1, 16), *range(16, 256, 16)]]
OLD_HELPERS = OLD_SWITCHES + [1 + 1024 * index for index in range(113, 128)]
NEW_HELPERS = NEW_SWITCHES + [1 + STRIDE * index for index in range(241, 256)]
LISTS = {}


def register(old, new):
    key = tuple(old)
    if key in LISTS:
        assert LISTS[key] == list(new)
    LISTS[key] = list(new)


register(OLD_SWITCHES, NEW_SWITCHES)
register(range(22), range(30))
register([1 + ((131072 - (value - 1)) % 131072) for value in OLD_SWITCHES],
         [1 + ((65536 - (value - 1)) % 65536) for value in NEW_SWITCHES])
register(range(36, 14, -1), range(44, 14, -1))
register(OLD_HELPERS, NEW_HELPERS)
register([(value - 1) // 1024 for value in OLD_HELPERS],
         [(value - 1) // STRIDE for value in NEW_HELPERS])
register([1 + 1024 * index for index in range(128)],
         [1 + STRIDE * index for index in range(256)])
register([1 + 1024 * (-index % 128) for index in range(128)],
         [1 + STRIDE * (-index % 256) for index in range(256)])
register([(index - (member - 1) // 1024) % 128
          for index in range(128) for member in OLD_HELPERS],
         [(index - (member - 1) // STRIDE) % 256
          for index in range(256) for member in NEW_HELPERS])
register([1] + [0] * 127, [1] + [0] * 255)
register([15 + leaf // 15 for leaf in range(105)],
         [15 + leaf // 15 for leaf in range(225)])
register([leaf % 15 for leaf in range(105)], [leaf % 15 for leaf in range(225)])
register([1 + 1024 * (16 * (leaf // 15 + 1)) for leaf in range(105)],
         [1 + STRIDE * (16 * (leaf // 15 + 1)) for leaf in range(225)])
register([1 + 1024 * (128 - 16 * (leaf // 15 + 1)) for leaf in range(105)],
         [1 + STRIDE * (256 - 16 * (leaf // 15 + 1)) for leaf in range(225)])
register([21 - leaf // 15 for leaf in range(105)],
         [29 - leaf // 15 for leaf in range(225)])
register([1 + 1024 * (leaf % 15 + 1 + 16 * (leaf // 15 + 1)) for leaf in range(105)],
         [1 + STRIDE * (leaf % 15 + 1 + 16 * (leaf // 15 + 1)) for leaf in range(225)])
register(range(15, 22), range(15, 30))
register(range(21, 14, -1), range(29, 14, -1))
register([1 + 1024 * (128 - 16 * (member + 1)) for member in range(7)],
         [1 + STRIDE * (256 - 16 * (member + 1)) for member in range(15)])
for slot in range(16):
    register([1 + 1024 * (16 * (member + 1) + slot) for member in range(7)],
             [1 + STRIDE * (16 * (member + 1) + slot) for member in range(15)])
for parent in range(15):
    register([15 * member + parent for member in range(7)],
             [15 * member + parent for member in range(15)])
schemas = list(dict.fromkeys(re.findall(r'\(\[\.bytes (\[[\d, ]+\])',
                                       (HERE / 'generated/Stage_prelude.lean').read_text())))
assert len(schemas) == 2
for module, name, new in [('PeSetup', 'schemaBytes', schemas[0]),
                          ('PePrelude', 'peSchema', schemas[1])]:
    old = re.search(rf'def {name} : Blob := (\[[\d, ]+\])',
                    (REFERENCE / f'{module}.lean').read_text())[1]
    register(map(int, re.findall(r'\d+', old)), map(int, re.findall(r'\d+', new)))


def adapt(text):
    protected = []
    old_positions = '[p, p + 15, p + 30, p + 45, p + 60, p + 75, p + 90]'
    if old_positions in text:
        protected.append('[' + ', '.join('p' if member == 0 else f'p + {15 * member}'
                                        for member in range(15)) + ']')
        text = text.replace(old_positions, 'DATEPROTECTED0END')

    def protect(match):
        value = match.group(0)
        if re.fullmatch(r'\[\s*\(?-?\d+\)?(?:\s*,\s*\(?-?\d+\)?)*\s*\]', value):
            numbers = tuple(map(int, re.findall(r'-?\d+', value)))
            if numbers in LISTS:
                value = "[" + ", ".join(map(str, LISTS[numbers])) + "]"
        protected.append(value)
        return f"DATEPROTECTED{len(protected) - 1}END"

    text = re.sub(r'\[\s*\(?-?\d+\)?(?:\s*,\s*\(?-?\d+\)?)*\s*\]', protect, text)
    def protect_site(match):
        protected.append(match[2])
        return match[1] + f'DATEPROTECTED{len(protected) - 1}END'

    text = re.sub(r'(\+\+ \[)(\d+)(?=,)', protect_site, text)
    text = text.replace('2 ^ 7', '2 ^ DATELOGWIDTH').replace('2 ^ 16', '2 ^ DATELOGN')
    text = text.replace('pow_left 16', 'pow_left DATELOGN')
    text = re.sub(r'\bFin 7\b', 'Fin 15', text)
    text = re.sub(r'([<≤] )7\b', r'\g<1>15', text)
    text = re.sub(r'Int.toNat 7\b', 'Int.toNat 15', text)
    text = text.replace('7 * scalarBound', '15 * scalarBound')
    text = text.replace('seven member lanes', 'fifteen member lanes')
    text = text.replace('seven first-depth leaf members', 'fifteen first-depth leaf members')
    text = re.sub(r'(x : Fin )4\b', r'\g<1>2', text)
    text = text.replace('(45 : Int)', '(31 : Int)')
    text = text.replace('45 + Int.ofNat', '31 + Int.ofNat')
    text = re.sub(r'(lutInner\d+ \(model := model\) \(tape := tape\) \(path := path\))',
                  r'\1 (params := params)', text)
    text = text.replace('namespace PeProof',
                        'set_option maxRecDepth 200000\n\nnamespace PeProof', 1)
    text = text.replace('⟨512 * i,', '⟨DATEHALFSTRIDE * i,')
    constants = {65536: BASE, 32768: CUTOFF, 1024: STRIDE, 131072: 65536,
                 44: DIGITS, 88: 2 * DIGITS, 3872: 1800, 15488: 3600,
                 85184: 54000, 22: SWITCHES, 37: HELPERS, 36: 44, 21: 29,
                 105: LEAVES, 128: WIDTH, 127: WIDTH - 1,
                 4736: WIDTH * HELPERS, 501: 295, 457: 265, 120: 248,
                 73534451498: KEY_CUTOFF}
    old_setup = (REFERENCE / 'PeSetup.lean').read_text()
    for name, value in [('Q', Q), ('P', P), ('PQ', PQ), ('Winv', pow(WIDTH, -1, Q))]:
        old = int(re.search(rf'abbrev {name} : Nat := (\d+)', old_setup)[1])
        constants[old] = value
    old_key = (REFERENCE / 'PeKeygen.lean').read_text()
    law = re.search(r'abbrev keyLaw .*?\.gaussian \((\d+) / (\d+)\)', old_key)
    constants[int(law[1])] = KEY_NUMERATOR
    constants[int(law[2])] = KEY_DENOMINATOR
    old_half = re.search(r'abbrev halfQ : Int := (\d+)',
                         (REFERENCE / 'PePrelude.lean').read_text())[1]
    constants[int(old_half)] = Q // 2
    text = re.sub(r'(?<![\w])\d+(?![\w])',
                  lambda match: str(constants.get(int(match[0]), int(match[0]))), text)
    text = text.replace('DATELOGWIDTH', '8').replace('DATELOGN', '15')
    text = text.replace('DATEHALFSTRIDE', '128')
    text = text.replace('abbrev N : Nat := 4096', 'abbrev N : Nat := 32768')
    for index, value in enumerate(protected):
        value = re.sub(r'\d{20,}', lambda match: str(constants.get(int(match[0]), int(match[0]))),
                       value)
        text = text.replace(f'DATEPROTECTED{index}END', value)
    return text


def pre_wire(index):
    step = 0 if index < 145 else 1
    offset = 0 if step == 0 else 139
    local = index - offset
    selected = {25: 17, 26: 18, 27: 19, 30: 20, 31: 21, 32: 22,
                35: 23, 36: 24, 37: 25, 40: 26, 41: 27, 42: 28}
    if local < 18:
        return local + (125 if step else 0)
    if local in selected:
        return selected[local] + (125 if step else 0)
    return index - 14 * (step + 1)


def mapped_wire(stage, index):
    return index - 2 if stage == 'online' else pre_wire(index)


def rename_roots(text, stage, reverse=False):
    mapping = {index: mapped_wire(stage, index) for index in range(6 if stage == 'online' else 0, 284)}
    if stage == 'pre':
        mapping = {index: value for index, value in mapping.items()
                   if (index if index < 145 else index - 139) < 18 or
                   (index if index < 145 else index - 139) in
                   {25, 26, 27, 30, 31, 32, 35, 36, 37, 40, 41, 42} or
                   (index if index < 145 else index - 139) >= 43}
    if reverse:
        mapping = {value: key for key, value in mapping.items()}
    def repl(match):
        return match[1] + str(mapping.get(int(match[2]), int(match[2])))
    return re.sub(r'(generatedRoot(?:\.w_|_))(\d+)', repl, text)


def reference_facts(stage):
    module = 'Stage_online_eval' if stage == 'online' else 'Stage_preprocess'
    text = (HERE / 'generated' / f'{module}_Facts.lean').read_text()
    text = translate(text, module, reverse=True)
    text = rename_roots(text, stage, reverse=True)
    root_start = text.index('structure generatedRoot.Facts_0 ')
    head, root = text[:root_start], text[root_start:]
    names = rename_roots(' '.join(f'generatedRoot.w_{index}' for index in range(284)), stage)
    mapping = {int(value): index for index, value in enumerate(re.findall(r'\.w_(\d+)', names))}
    root = re.sub(r'(?<=\bh)(\d+)(?=_)',
                  lambda match: str(mapping.get(int(match[0]), int(match[0]))), root)
    return head + root


def write_lean(path, text, stage=None, root=False, protected=None):
    text = adapt(text)
    if stage:
        text = rename_roots(text, stage)
        if root:
            text = re.sub(r'((?:F|G)\.c\d+\.h)(\d+)(?=_)',
                          lambda match: match[1] + str(mapped_wire(stage, int(match[2]))), text)
            text = re.sub(r'(\[(?:5|7)\] \+\+ \[)(\d+)',
                          lambda match: match[1] + str(mapped_wire(stage, int(match[2]))), text)
    path = Path(path)
    for key, value in (protected or {}).items():
        text = text.replace(key, value)
    path.write_text(translate(text))


def write_decode(path, text, protected=None):
    text = adapt(text)
    text = re.sub(r'((?:generatedRoot(?:\.w_|_)|\.h))(\d+)',
                  lambda match: match[1] + str(int(match[2]) + 512
                                             if 521 <= int(match[2]) <= 560
                                             else int(match[2])), text)
    text = re.sub(r'(\+\+ \[)(\d+)(?=,)',
                  lambda match: match[1] + str(int(match[2]) + 512
                                             if 521 <= int(match[2]) <= 560
                                             else int(match[2])), text)
    facts = (HERE / 'generated/Stage_decode_Facts.lean').read_text()
    chunks = {}
    for match in re.finditer(r'^structure generatedRoot\.Facts_(\d+) .*?(?=^structure|^theorem)',
                             facts, re.M | re.S):
        for name in re.findall(r'^  (h\d+_\d+|hout) :', match[0], re.M):
            chunks[name] = match[1]
    text = re.sub(r'F\.c\d+\.(h\d+_\d+|hout)',
                  lambda match: 'F.c' + chunks[match[1]] + '.' + match[1], text)
    for key, value in (protected or {}).items():
        text = text.replace(key, value)
    Path(path).write_text(translate(text))


def generate_math():
    modules = ['PeSetup', 'PeDigits', 'PeCompanions', 'PeProduct', 'PeProductGlue',
               'PeStepRel', 'PeBranch', 'PeStepSum', 'PeStepData', 'PeLut',
               'PeOnlineBase', 'PeOnlineGlue', 'PePreBase', 'PePreGlue', 'PePreRootLayout',
               'PePrelude', 'PeEncrypt', 'PeKeygen', 'PeSlots', 'PePhase', 'PeAnchor',
               'PeDecrypt', 'PeDecryptBits', 'PeKeyLink', 'PeDecode', 'PeDecodeLink']
    for module in modules:
        template = HERE / f'{module}.lean.in'
        if module in {'PePrelude', 'PeEncrypt', 'PeKeygen', 'PeSlots',
                      'PeDecode', 'PeDecodeLink', 'PeKeyLink'} and template.exists():
            (HERE / f'{module}.lean').write_text(template.read_text())
            continue
        source = (REFERENCE / f'{module}.lean').read_text()
        text = adapt(source)
        if module == 'PeSetup':
            start = text.index('/-!')
            finish = text.index('-/', start) + 2
            text = text[:start] + '''/-!
# The date parameter set

The ring `Z_Q[X]/(X^32768 + 1)` uses 15 24-bit CRT primes (`log2 Q ≈ 354.97`).
The gadget base is `2^12`, with two digits per limb, so `D = 30` and the joined row has
60 columns. The lookup group has order 256 and a two-level radix-16 plan: 15 internal
branches, 15 first-depth leaves, and 225 second-depth leaves. Each leaf member shares
its companion decompositions across groups of at most four coefficient slots.

Compact companions use `R_PQ`, with 15 extension primes (`log2 P ≈ 338.85`).
The statistical TrapGen anchor has 265 sampled columns plus the 30 gadget columns.
-/''' + text[finish:]
        if module in {'PePrelude', 'PeEncrypt', 'PeProductGlue'}:
            text = re.sub(r'\bFin 4\b', 'Fin 2', text)
        if module == 'PeLut':
            text = text.replace('H = ⟨1025⟩', 'H = ⟨257⟩')
            text = text.replace('lut_fourier_table_value 7 N 512',
                                'lut_fourier_table_value 8 N 128')
            text = text.replace('Fin 8', 'Fin 16')
        if module == 'PeOnlineBase':
            text = text.replace('(x : Fin 4)', '(x : Fin 2)')
        if module == 'PeSlots':
            text = text.replace('8 * a * b', 'a * b')
        if module == 'PePreGlue':
            start = text.index('theorem select_four')
            finish = text.index('theorem familyStatic_four', start)
            text = text[:start] + '''theorem select_two {α : Type} {a b o : α} {idx : Int} {p : Fin 2}
    (h : select idx [a, b] o) (hp : idx = ((p : Nat) : Int)) : o = ![a, b] p := by
  obtain ⟨q, hq, ho⟩ := h
  have hqp : (q : Nat) = p := by omega
  rw [ho]
  obtain ⟨q, hq'⟩ := q
  obtain ⟨p, hp'⟩ := p
  simp only at hqp
  subst hqp
  interval_cases q <;> rfl

''' + text[finish:]
            text = text.replace('familyStatic_four {f : Fin 4', 'familyStatic_four {f : Fin 2')
            text = text.replace('(hk : k < 4)', '(hk : k < 2)')
        if module in {'PeDecode', 'PeDecodeLink'}:
            write_decode(HERE / f'{module}.lean', source)
        else:
            (HERE / f'{module}.lean').write_text(text)


def generate_requirement():
    text = (REFERENCE / 'PeRequirement.lean').read_text()
    text = text.replace('[9]', '[10]')
    start = text.index('theorem iter_four')
    finish = text.index('theorem and_step', start)
    text = text[:start] + '''theorem iter_two {σ : Type} {body : Nat → σ → σ → Prop} {init final : σ}
    (h : MxxIR.IterRuns body (Int.toNat 2) init final) :
    ∃ middle, body 0 init middle ∧ body 1 middle final := by
  obtain ⟨c1, h1, b1⟩ := MxxIR.IterRuns.step_of_succ (count := 1) h
  obtain ⟨c0, h0, b0⟩ := MxxIR.IterRuns.step_of_succ (count := 0) h1
  cases h0
  exact ⟨c1, b0, b1⟩

''' + text[finish:]
    text = re.sub(r'\bFin 4\b', 'Fin 2', text)
    text = re.sub(r'\bFin 5\b', 'Fin 3', text)
    text = text.replace('v 4', 'v 2').replace('v_4', 'v_2').replace('< 4', '< 2')
    text = text.replace('384', '256').replace('128', '256')
    text = text.replace('first three steps', 'first step')
    text = text.replace('∀ hi : i < 512,', '∀ hi : i < 256,')
    text = text.replace('∀ hi : i < 256, t ⟨i, hi⟩',
                        '∀ hi : i < 256, t ⟨i, by omega⟩')
    text = text.replace('h6 hw6).2 j hj j.isLt', 'h6 hw6).2 j hj hj')
    text = text.replace('⟨s1, s2, s3, b0, b1, b2, b3⟩ := iter_four',
                        '⟨s1, b0, b1⟩ := iter_two')
    text = text.replace('![0, s1, s2, s3, w.w_21_0]', '![0, s1, w.w_21_0]')
    text = text.replace('    · exact step 2 (by norm_num) _ _ b2\n', '')
    text = text.replace('    · exact step 3 (by norm_num) _ _ b3\n', '')
    (HERE / 'PeRequirement.lean').write_text(text)


def write_chain(path, text):
    text = adapt(text)
    for module, stage in [('Stage_online_eval', 'online'), ('Stage_preprocess', 'pre')]:
        text = re.sub(rf'{module}\.generatedRoot(?:\.w_|_)\d+',
                      lambda match: rename_roots(match[0], stage), text)
    text = re.sub(r'(G\.c\d+\.h)(\d+)(?=_)',
                  lambda match: match[1] + str(pre_wire(int(match[2]))), text)
    Path(path).write_text(translate(text))


def generate_certificate():
    template = HERE / 'PeCertificate.lean.in'
    if template.exists():
        (HERE / 'PeCertificate.lean').write_text(template.read_text())
        return
    text = (REFERENCE / 'PeCertificate.lean').read_text()
    text = text.replace('G.c26.hout', 'G.c9.hout').replace('F.c10.hout', 'F.c5.hout')
    text = text.replace('K.c12.hout', 'K.c24.hout')
    text = text.replace('chain_all F G rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl',
                        'chain_all F G rfl rfl rfl rfl rfl rfl')
    start = text.index('    rfl rfl rfl rfl\n    (leafFamilies_eta (fun slot ↦ preLeafT_2')
    finish = text.index('    (eB :=', start)
    text = text[:start] + '    hs\n' + text[finish:]
    for old, virtual in [(540, 264), (546, 270), (552, 276), (558, 282)]:
        text = text.replace(f'w_{old}_0', f'w_{virtual}_0')
    text = text.replace(REFERENCE.name, HERE.name)
    text = text.replace('Eq.trans (by fin_cases i <;> rfl) (hanc i)',
                        'Eq.trans (by rfl) (hanc i)')
    text += '''
namespace PeCertificate

/-- The exported date correctness claim, also available as `PeProof.certificate`. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := PeProof.certificate

end PeCertificate
'''
    write_chain(HERE / 'PeCertificate.lean', text)


def generate_numeric():
    companion = (31 * (Q // 2) + (N + 2) * (P // 2)) // P
    rounded = (31 + (N + 2) * (P // 2)) // P
    def product(bound):
        return 60 * N * bound * CUTOFF + 30 * (60 * N * companion * CUTOFF) + 31
    def switch(bound):
        return 60 * N * 31 * CUTOFF + product(bound)
    def scalar(bound):
        return 30 * N * bound * CUTOFF
    def step(bound):
        product_bound = product(bound)
        return (scalar(product_bound) + 30 * scalar(switch(product_bound)) +
                225 * scalar(switch(switch(product_bound))))
    final = step(step(31))
    error = (256 * 295 * N * rounded * KEY_CUTOFF +
             256 * 30 * N * companion * KEY_CUTOFF +
             256 * N * KEY_CUTOFF + 11520 * 60 * N * 31 * KEY_CUTOFF +
             60 * N * final * KEY_CUTOFF)
    assert 4 * (31 + error + 1) + 4 < Q
    text = f'''import PeDecryptBits
import PeOnlineGlue

namespace PeProof

/-- The exact inherited error bound after one online step. -/
def Bstep (B : Nat) : Nat :=
  stepBound (productBound B companionBound) (bound1 (productBound B companionBound))
    (bound2 (productBound B companionBound))

/-- The exact output bound after the two date lookups. -/
def Bfinal : Nat := Bstep (Bstep 31)

set_option maxRecDepth 100000 in
theorem margin : 4 * (31 + decErr Bfinal + 1) + 4 < Q := by
  have hc : companionBound = {companion} := by unfold companionBound; norm_num
  have hr : roundBound = {rounded} := by unfold roundBound; norm_num
  unfold Bfinal Bstep stepBound productBound bound1 bound2 switchBound scalarBound decErr keyCutoff
  rw [hc, hr]
  decide +kernel

end PeProof
'''
    (HERE / 'PeNumeric.lean').write_text(text)


if __name__ == '__main__':
    generate_math()
    generate_requirement()
    generate_numeric()
    generate_certificate()
