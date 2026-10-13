#!/bin/bash
# The refresh_top proof pipeline, end to end, for a worst-case parameter directory $1 (its
# `generated/` export already written, facts on) and optionally its average-case package $2:
#
#   1. roles and adapters      gen_adapters.py (roles from xroles.py; bundles into GenIface.lean)
#   2. instance / part theorem $INSTANCE_SCRIPTS (package D; each run as `python3 <script> $1`)
#   3. looped decode           gen_decode_loops.py      -> TopDecodeLoops.lean
#   4. requirement facts       gen_requirement2.py      -> Gen_Requirement.lean
#   5. correctness             gen_correct2.py          -> Gen_Correct.lean (part theorem $PART =
#                              `Module:theorem`; without it, a `sorry` placeholder `part_terms`)
#   6. average bridge          gen_bridge.py            -> $2/worst, $2/bridge
#   7. port                    gen_port.py              -> $2/proof (the modules below, renamed)
#   8. average certificate     gen_correct2.py --certificate -> $2/proof/RefreshTopCertificate.lean,
#                              and the `RefreshTopProof` roots of $2/lakefile.toml
#   9. (BUILD=1) `lake build RefreshTopCertificate` in $2 (ulimit -v 41943040).
#
# Environment: PART, INSTANCE_SCRIPTS (space-separated generator scripts of the instance/part
# pipeline), PART_MODULES (the modules they write, ported to $2 in dependency order).
set -e
d=$1
avg=$2
here=$(cd "$(dirname "$0")" && pwd)

# 1. roles and adapters (stages with roles; the bundles go into GenIface.lean)
python3 "$here/gen_adapters.py" "$d/generated" "$d" Stage_preprocess Stage_encrypt Stage_online_eval \
  Stage_decode --iface "$d/GenIface.lean"

# 2. instance / part theorem (package D)
for s in $INSTANCE_SCRIPTS; do
  python3 "$here/$s" "$d"
done

# 3.–5. decode, requirement, correctness
python3 "$here/gen_decode_loops.py" "$d"
python3 "$here/gen_requirement2.py" "$d"
if [ -n "$PART" ]; then
  python3 "$here/gen_correct2.py" "$d" --part "$PART"
else
  python3 "$here/gen_correct2.py" "$d"
fi

[ -z "$avg" ] && exit 0

# 6.–8. the average-case package
python3 "$here/gen_bridge.py" "$avg" "$d"
GENERIC="TopCrtGeneric GenIface GenDecode GenReq"
ADAPTERS=""
for m in Stage_preprocess Stage_encrypt Stage_online_eval Stage_decode; do
  [ -n "$PART_MODULES" ] && [ -f "$d/Gen_${m}_Scopes.lean" ] && ADAPTERS="$ADAPTERS Gen_${m}_Scopes"
done
MODULES="$GENERIC $ADAPTERS $PART_MODULES Gen_Requirement TopDecodeLoops Gen_Correct"
python3 "$here/gen_port.py" "$avg" "$d" $MODULES
if [ -n "$PART" ]; then
  python3 "$here/gen_correct2.py" "$avg" --certificate --guard
else
  python3 "$here/gen_correct2.py" "$avg" --certificate
fi
python3 - "$avg/lakefile.toml" $MODULES RefreshTopCertificate <<'EOF'
import re
import sys
path, mods = sys.argv[1], sys.argv[2:]
s = open(path).read()
i = s.index('name = "RefreshTopProof"')
m = re.compile(r'roots = \[(.*?)\]').search(s, i)
roots = [x.strip().strip('"') for x in m.group(1).split(',') if x.strip()]
roots += [x for x in mods if x not in roots]
s = s[:m.start()] + 'roots = [' + ', '.join(f'"{x}"' for x in roots) + ']' + s[m.end():]
open(path, 'w').write(s)
EOF

# 9. build
if [ -n "$BUILD" ]; then
  (cd "$avg" && ulimit -v 41943040 && lake build RefreshTopCertificate)
fi
