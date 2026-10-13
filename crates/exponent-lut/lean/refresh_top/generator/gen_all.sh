#!/bin/bash
# Write every proof file of part $2 of parameter directory $1, in dependency order: the
# hand-written templates of part `0_0` (renamed), then every generator.
set -e
d=$1
part=$2
here=$(dirname "$0")
for f in Mask MaskCarrier Fresh FreshCarrier; do
  sed "s/0_0/${part}/g" "$d/Part_0_0_$f.lean" > "$d/Part_${part}_$f.lean"
done
for g in gen_cells gen_sums gen_pre_sums gen_enc_root gen_enc gen_enc_facts gen_pre_rhs gen_pre_switch gen_root_luts \
         gen_online_luts gen_online_root gen_root_sums gen_online_chain gen_smudge_members gen_smudge_totals \
         gen_smudge_chain gen_keys_cells gen_key_facts gen_keys_luts gen_keys_root gen_inputs gen_pre_place \
         gen_key_links gen_fresh_chain gen_labels gen_online_final gen_part_theorem; do
  PARTS=$part python3 "$here/$g.py" "$d" > /dev/null
done
