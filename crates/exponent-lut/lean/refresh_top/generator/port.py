"""Run proof generators on the merged export through the per-part views (`view.py`).

`python3 port.py <parameter directory> <generator> ...` prepares `<d>/.port` (the stage views as
`generated/Stage_<phase>_I.lean`, links to the other generated modules and to the hand-written
proof files), runs each generator there for part `I`, and writes every `Part_I_*.lean` it produced
to `<d>`, mapped onto the merged stages (`view.realize`).
"""
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import view

here = os.path.dirname(os.path.abspath(__file__))


def prepare(d):
    port = f'{d}/.port'
    os.makedirs(f'{port}/generated', exist_ok=True)
    for f in glob.glob(f'{d}/generated/*.lean'):
        link = f'{port}/generated/{os.path.basename(f)}'
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(f, f'{port}/generated'), link)
    for p in view.PHASES:
        view.make_view(d, p)
        os.replace(f'{d}/view/Stage_{p}_I.lean', f'{port}/generated/Stage_{p}_I.lean')
    os.rmdir(f'{d}/view')
    for f in glob.glob(f'{d}/*.lean'):
        name = os.path.basename(f)
        if name.startswith('Part_I_'):
            continue
        link = f'{port}/{name}'
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(f, port), link)
    return port


def realize(d, port):
    texts = {f: view.realize(open(f).read(), d) for f in sorted(glob.glob(f'{port}/Part_I_*.lean'))}
    # definitions taking the instance make their users take it: iterate to the fixpoint
    while True:
        names = view.instance_defs(texts.values())
        new = {f: view.bind_instance(view.apply_instance(text, names)) for f, text in texts.items()}
        if new == texts:
            break
        texts = new
    for f, text in texts.items():
        name = os.path.basename(f)[len('Part_I_'):-len('.lean')]
        if name in LOCAL_USERS and 'import Part_I_Local' not in text:
            # the decoding ring's lemmas (`gen_local.py`) shadow the shared ones
            text = 'import Part_I_Local\n' + text
        if name == 'Assembly':
            # ... by their full names: a bare name is ambiguous with the shared lemma, whose
            # elaboration against the instance's ring does not terminate
            text = qualify_local(text, d)
        open(f'{d}/{os.path.basename(f)}', 'w').write(text)
    roots(d)


def roots(d):
    """List the proof's modules (the import closure of `TopCorrect`) as the roots of its library
    `RefreshTopProof`."""
    import re
    seen, todo = set(), ['TopCorrect']
    while todo:
        m = todo.pop()
        if m in seen or not os.path.exists(f'{d}/{m}.lean'):
            continue
        seen.add(m)
        todo += re.findall(r'^import (\S+)', open(f'{d}/{m}.lean').read(), re.M)
    path = f'{d}/lakefile.toml'
    text = open(path).read()
    head = text.index('name = "RefreshTopProof"')
    a = text.index('roots = [', head)
    b = text.index(']', a)
    text = text[:a] + 'roots = [' + ', '.join(f'"{n}"' for n in sorted(seen)) + text[b:]
    open(path, 'w').write(text)


# The modules that use the shared lemmas of the decoding ring (`Part_I_Local`).
LOCAL_USERS = {'Inputs', 'FreshChain', 'OnlineFinal', 'Assembly', 'OnlineKey'}

# The hand-written proofs of part `0_0` that `gen_all.sh` instantiates for every part.
TEMPLATES = ['Mask', 'MaskCarrier', 'Fresh', 'FreshCarrier']


def instance_rings(text):
    """Chain `0`'s decoding ring and dropped prime as the instance's (`R i0`, `qd i0`)."""
    import re
    text = re.sub(r'\bQ1\b', '(R i0)', text)
    text = re.sub(r'\bq3\b', '(qd i0)', text)
    text = re.sub(r'\bq0_eq\b', '(q0_eqR i0 hi0)', text)
    # numeric side conditions next to the instance's ring identity
    lines = text.split('\n')
    for k, line in enumerate(lines):
        if 'q0_eqR i0 hi0' in line or (k and 'q0_eqR i0 hi0' in lines[k - 1]
                                        and line.lstrip().startswith('(by norm_num)')):
            lines[k] = line.replace('(by norm_num)', '(by inst_num)')
    text = '\n'.join(lines)
    return text.replace('import TopLemmas\n', 'import TopLemmas\nimport TopInst\n', 1)


def remap_nodes(text, stage):
    """Identifiers ending in a node id of the old part-`0_0` stage `stage` (scopes, lemma and
    hypothesis names) with the merged body's node id (`maps/<stage>.json`, from the checked
    isomorphism of the two graphs)."""
    import json
    import re
    nodes = {int(k): v for k, v in json.load(open(f'{here}/maps/{stage}.json'))['nodes'].items()}
    return re.sub(r'(?<=[A-Za-z_])(\d{3,})\b(?!_)',
                  lambda m: str(nodes.get(int(m.group(1)), int(m.group(1)))), text)


def seed(d, port, name):
    """Instantiate the template `Part_0_0_<name>.lean` for part `I` (its ported copy
    `template/<name>.lean` when there is one, else with `instance_rings`)."""
    src = f'{here}/template/{name}.lean'
    if os.path.exists(src):
        text = open(src).read()
    else:
        text = instance_rings(open(f'{d}/Part_0_0_{name}.lean').read())
    open(f'{port}/Part_I_{name}.lean', 'w').write(text.replace('0_0', 'I'))


if __name__ == '__main__':
    d = sys.argv[1]
    port = prepare(d)
    env = dict(os.environ, PARTS='I')
    for g in sys.argv[2:]:
        if g in TEMPLATES:
            seed(d, port, g)
        else:
            subprocess.run([sys.executable, f'{here}/{g}.py', port], env=env, check=True)
    realize(d, port)
