"""The Lake package of a parameter set: `lakefile.toml`, `lake-manifest.json`, and `lean-toolchain`."""
import json
import os

from gen_blocks import TEMPLATES
from info import wrap


def package_name(I):
    pid = os.path.basename(os.path.abspath(I.d))
    return 'MxxExponentLutLwr' + ''.join(p[:1].upper() + p[1:] for p in pid.split('-'))


def modules(I):
    return (['LwrSetup', 'LwrInputs', 'LwrEncrypt', 'LwrPreprocess', 'LwrCells'] +
            [f'LwrBlock{b}' for b in range(I.B)] + ['LwrLut', 'LwrCertificate'])


def h2_modules(I):
    """The modules of the certificate under H2: the errors named, the model, the decomposition and
    its bounds."""
    return (['LwrEncryptE', 'LwrPreprocessE', 'LwrErrorCommon'] + [f'LwrBlockE{b}' for b in range(I.B)] +
            ['LwrLutE', 'LwrModelBase'] + [f'LwrModel{b}' for b in range(I.B)] +
            ['LwrModelLut', 'LwrChainAll', 'LwrChainFinal', 'LwrH2Model', 'LwrH2Bounds', 'LwrH2Agree',
             'LwrH2Cert'])


def lakefile(I):
    roots = wrap([f'"{m}"' for m in modules(I)], ', ', 2, 2)
    return f'''name = "{package_name(I)}"
version = "0.1.0"
defaultTargets = ["LwrGenerated", "LwrProof"{', "LwrH2"' if I.h2 else ''}]

[leanOptions]
relaxedAutoImplicit = false
pp.unicode.fun = true

[[require]]
name = "MxxExponentLutLwr"
path = ".."

# Modules written by `lwr_prf_end_to_end` (`tests/lwr_prf.rs`) for this parameter set.
[[lean_lib]]
name = "LwrGenerated"
srcDir = "generated"
roots = [
  "Backend", "Claim", "Ideal", "Requirement_0", "Stage_payload", "Stage_mask",
  "Stage_preprocess", "Stage_encrypt", "Stage_online_eval", "Stage_decode"
]

# The proof of the generated claim, written by `../generator/gen.py`. The certificate chains the
# {I.B} blocks in one proof, whose kernel check needs a larger thread stack than the default.
[[lean_lib]]
name = "LwrProof"
moreLeanArgs = ["--tstack=512000"]
roots = [
{roots}
]
''' + ('' if not I.h2 else f'''
# The certificate under H2: the errors named (`*E`), the model of a run (`LwrModel*`, `LwrChain*`),
# and the decomposition of the decoded coefficient with its bounds (`LwrH2*`).
[[lean_lib]]
name = "LwrH2"
moreLeanArgs = ["--tstack=4000000"]
roots = [
{wrap([f'"{m}"' for m in h2_modules(I)], ', ', 2, 2)}
]
''')


def manifest(I):
    """The manifest of the template parameter set, under this package's name: every parameter set
    has the same dependencies at the same relative paths."""
    template = os.path.join(TEMPLATES, 'lake-manifest.json')
    text = open(template).read()
    name = json.loads(text)['name']
    return text.replace(f'"name": "{name}"', f'"name": "{package_name(I)}"', 1)


def toolchain(I):
    return open(os.path.join(os.path.dirname(os.path.abspath(I.d)), 'lean-toolchain')).read()
