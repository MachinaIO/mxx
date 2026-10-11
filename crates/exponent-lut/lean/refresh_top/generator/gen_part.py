"""Instantiate the proof templates of `templates/` for every part of parameter directory `d`.

A template is the proof of part `(c, j)` with markers:

- `@P@`: the part suffix `c_j`; `@E@`, `@O@`, `@R@`: the stage namespaces
  `Stage_encrypt_c_j`, `Stage_online_eval_c_j`, `Stage_preprocess_c_j`.
- `@DESTRUCT(stage, def, hyp)@`: the destructuring tactics of `def` (a definition of stage file
  `stage` in `{encrypt, online_eval, preprocess}`), see `patterns.destruct_inline`; with `def`
  `generatedRoot`, of the root body (`patterns.destruct`).
- `@TAG(stage, def, atom)@`: the hash tag prefix of atom `atom` of `def`.
- `@INPUT(stage, wire)@`: the projection of root input `wire` of `stage`; `@OUTPUT(stage, wire)@`:
  the projection of output `wire` (`outputs.2...1`).

Usage: `python3 gen_part.py <parameter directory> [template ...]`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from patterns import destruct, destruct_inline, parse, parse_inline

here = os.path.dirname(os.path.abspath(__file__))
d = None


def parts_of(directory):
    if os.environ.get('PARTS'):
        return os.environ['PARTS'].split(',')
    return sorted(re.match(r'Stage_online_eval_(\d_\d)\.lean', f).group(1)
                  for f in os.listdir(f'{directory}/generated')
                  if re.match(r'Stage_online_eval_\d_\d\.lean', f))


def stage_file(stage, part):
    return f'{d}/generated/Stage_{stage}_{part}.lean'


def atoms_of(path, name):
    if name == 'generatedRoot':
        _, n, a = parse(path, name)
    else:
        _, n, a = parse_inline(path, name)
    return dict(zip(n, a))


def root_defs(path):
    text = open(path).read()
    body = text[text.index('abbrev generatedRoot.body'):]
    return dict(re.findall(r'^  let (w_\S+) : [^\n]*?:= (.*)$', body, re.M))


def instantiate(template, part):
    out = template.replace('@P@', part)
    for key, stage in (('@E@', 'encrypt'), ('@O@', 'online_eval'), ('@R@', 'preprocess')):
        out = out.replace(key, f'Stage_{stage}_{part}')

    def destruct_marker(m):
        stage, name, hyp = (x.strip() for x in m.group(1).split(','))
        path = stage_file(stage, part)
        ns = f'Stage_{stage}_{part}'
        if name == 'generatedRoot':
            return destruct(path, ns, name, hyp).rstrip('\n')
        return destruct_inline(path, ns, name, hyp).rstrip('\n')

    def tag_marker(m):
        stage, name, atom = (x.strip() for x in m.group(1).split(','))
        at = atoms_of(stage_file(stage, part), name)[atom]
        return re.search(r'hashSample \(hashModel\) \(\[([\d, ]+)\]\)', at).group(1)

    def input_marker(m):
        stage, wire = (x.strip() for x in m.group(1).split(','))
        return root_defs(stage_file(stage, part))[wire]

    def output_marker(m):
        stage, wire = (x.strip() for x in m.group(1).split(','))
        at = atoms_of(stage_file(stage, part), 'generatedRoot')['hout']
        outs = re.findall(r'\b(w_\d+_\d+)\b', at)
        i = outs.index(wire)
        return 'outputs' + '.2' * i + ('.1' if i < len(outs) - 1 else '')

    out = re.sub(r'@DESTRUCT\(([^)]*)\)@', destruct_marker, out)
    out = re.sub(r'@TAG\(([^)]*)\)@', tag_marker, out)
    out = re.sub(r'@INPUT\(([^)]*)\)@', input_marker, out)
    out = re.sub(r'@OUTPUT\(([^)]*)\)@', output_marker, out)
    return out


if __name__ == '__main__':
    d = sys.argv[1]
    names = sys.argv[2:] or sorted(f[:-len('.lean.tpl')] for f in os.listdir(f'{here}/templates')
                                   if f.endswith('.lean.tpl'))
    parts = parts_of(d)
    for name in names:
        template = open(f'{here}/templates/{name}.lean.tpl').read()
        for part in parts:
            open(f'{d}/Part_{part}_{name}.lean', 'w').write(instantiate(template, part))
    print(f'instantiated {names} for parts {parts}')
