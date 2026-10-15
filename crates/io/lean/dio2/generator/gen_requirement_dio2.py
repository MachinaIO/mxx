#!/usr/bin/env python3
"""Write `proof/Dio2Requirement.lean`: the requirement facts of the dio2 claim.

Runs refresh_top's `gen_requirement2.py` on the worst-case module `W_Requirement_0` of an
average-case package (namespace `Dio2Requirement`). The root conditions of the dio2 requirement are
conjunctions the generator states unsplit, as sums of `if … then 1 else 0`; their numerals are
typed `Int`, as in the export.

Usage: gen_requirement_dio2.py <average package>
"""
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
pkg = sys.argv[1]
out = os.path.join(pkg, 'proof', 'Dio2Requirement.lean')
subprocess.run([sys.executable, os.path.join(here, '../../../../exponent-lut/lean/refresh_top/generator/gen_requirement2.py'),
                pkg, '--gdir', os.path.join(pkg, 'worst'), '--module', 'W_Requirement_0',
                '--namespace', 'Dio2Requirement', '--out', out], check=True)
text = open(out).read()
text = re.sub(r'^(  cond_\d+ : .*)$', lambda m: m.group(1).replace('then 1 else 0)', 'then (1 : Int) else 0)'),
              text, flags=re.M)
open(out, 'w').write(text)
