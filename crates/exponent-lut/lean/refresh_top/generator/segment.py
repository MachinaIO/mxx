"""Segments of a large generated definition: a theorem over the atoms of a chosen part of the
definition, with the wires it mentions as implicit variables.

`segment(path, ns, name, atoms)` returns `(binders, call)`: the Lean binders of the wires
(`{w : T}`), the computed `let`s (`(e_w : w = value)`, recursively) and the atoms (`(h : atom)`),
and the arguments that apply the theorem inside a proof that destructured `name` with
`patterns.destruct_root` (atoms by name, `let` equations by `rfl`).
"""
import re

from patterns import _block, parse

WIRE = re.compile(r'\bw_\d+_(?:\d+|concat_\d+)\b')


def abbreviations(setup_text):
    return {v: k for k, v in re.findall(r'abbrev (\w+) : Nat := (\d+)', setup_text)}


def short(t, abbrevs):
    t = t.replace('Mxx.Primitives.', '')
    return re.sub(r'\b(\d{6,})\b', lambda m: abbrevs.get(m.group(1), m.group(1)), t)


def qualify(text, ns):
    text = re.sub(r'(?<![\w.])((?:parallel_|sequential_)+generatedRoot_\w+)', rf'{ns}.\1', text)
    text = re.sub(r'\bbackend\b', 'Backend.backend', text)
    text = re.sub(r'\bhashModel\b', 'model', text)
    return text.replace('MxxRuntime.', '').replace('Mxx.Primitives.', '')


def segment(path, ns, name, wanted, abbrevs, extra=()):
    lets, names, atoms = parse(path, name)
    atom = dict(zip(names, atoms))
    lines = open(path).read().split('\n')
    body = _block(lines, 'abbrev ' + name + '.body')
    types, values = {}, {}
    for line in body:
        m = re.match(r'let (\S+) : (.*?) := (.*)$', line)
        if m:
            types[m.group(1)] = m.group(2)
            if not m.group(3).startswith('inputs') and not m.group(3).startswith('witness'):
                values[m.group(1)] = m.group(3)
    wit = _block(lines, f'structure {name}.Witness')
    for line in wit:
        m = re.match(r'(\S+) : (.*)$', line)
        if m:
            types[m.group(1)] = m.group(2)
    wires, eqs, todo = [], [], list(extra)
    for a in wanted:
        todo += WIRE.findall(atom[a])
    seen = set()
    while todo:
        x = todo.pop(0)
        if x in seen:
            continue
        seen.add(x)
        wires.append(x)
        if x in values:
            eqs.append(x)
            todo += WIRE.findall(values[x])
    order = {w: i for i, w in enumerate(lets)}
    wires.sort(key=lambda w: order.get(w, 10 ** 6))
    eqs.sort(key=lambda w: order.get(w, 10 ** 6))
    binders = [f'{{{w} : {short(types[w], abbrevs)}}}' for w in wires]
    binders += [f'(e_{w[2:]} : {w} = {short(qualify(values[w], ns), abbrevs)})' for w in eqs]
    binders += [f'({a} : {short(qualify(atom[a], ns), abbrevs)})' for a in wanted]
    call = ' '.join(wanted) + ' ' + ' '.join(f'(e_{w[2:]} := rfl)' for w in eqs)
    return binders, call
