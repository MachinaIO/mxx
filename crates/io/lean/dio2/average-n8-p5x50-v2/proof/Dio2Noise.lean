import Mathlib.Tactic.NormNum

set_option maxRecDepth 16384
set_option maxHeartbeats 8000000

namespace Dio2NoiseCertificate

def n : Nat := 8
theorem n_eval : n = 8 := by
  norm_num [n]

def m : Nat := 25
theorem m_eval : m = 25 := by
  norm_num [m]

def base : Nat := 4096
theorem base_eval : base = 4096 := by
  norm_num [base]

def anchor : Nat := 54
theorem anchor_eval : anchor = 54 := by
  norm_num [anchor]

def cutoff : Nat := 22011706
theorem cutoff_eval : cutoff = 22011706 := by
  norm_num [cutoff]

def half : Nat := 2048
theorem half_eval : half = base / 2 := by
  norm_num [half, base]

def action : Nat := 819200
theorem action_eval : action = (m + m) * n * half := by
  norm_num [action, half, m, n]

def modulus : Nat := 1809251394327365744250316798798350396453362684746139641520435240552807944993
theorem modulus_eval : modulus = 1809251394327365744250316798798350396453362684746139641520435240552807944993 := by
  norm_num [modulus]

def prime0 : Nat := 1125899906842273
theorem prime0_eval : prime0 = 1125899906842273 := by
  norm_num [prime0]

def quotient0 : Nat := 1606938044254428791916078459060479968194539524630727034176641
theorem quotient0_eval : quotient0 = 1606938044254428791916078459060479968194539524630727034176641 := by
  norm_num [quotient0]

theorem quotient0_exact : quotient0 * prime0 = modulus := by
  norm_num [quotient0, prime0, modulus]

def prime1 : Nat := 1125899906842177
theorem prime1_eval : prime1 = 1125899906842177 := by
  norm_num [prime1]

def quotient1 : Nat := 1606938044254565807694577896670997561277593686540426466090209
theorem quotient1_eval : quotient1 = 1606938044254565807694577896670997561277593686540426466090209 := by
  norm_num [quotient1]

theorem quotient1_exact : quotient1 * prime1 = modulus := by
  norm_num [quotient1, prime1, modulus]

def prime2 : Nat := 1125899906842033
theorem prime2_eval : prime2 = 1125899906842033 := by
  norm_num [prime2]

def quotient2 : Nat := 1606938044254771331362327096896778694705812889646799470123121
theorem quotient2_eval : quotient2 = 1606938044254771331362327096896778694705812889646799470123121 := by
  norm_num [quotient2]

theorem quotient2_exact : quotient2 * prime2 = modulus := by
  norm_num [quotient2, prime2, modulus]

def prime3 : Nat := 1125899906841713
theorem prime3_eval : prime3 = 1125899906841713 := by
  norm_num [prime3]

def quotient3 : Nat := 1606938044255228050623992174508164557277769211170177223088561
theorem quotient3_eval : quotient3 = 1606938044255228050623992174508164557277769211170177223088561 := by
  norm_num [quotient3]

theorem quotient3_exact : quotient3 * prime3 = modulus := by
  norm_num [quotient3, prime3, modulus]

def prime4 : Nat := 1125899906841377
theorem prime4_eval : prime4 = 1125899906841377 := by
  norm_num [prime4]

def quotient4 : Nat := 1606938044255707605848740785410594412685698358681526360823809
theorem quotient4_eval : quotient4 = 1606938044255707605848740785410594412685698358681526360823809 := by
  norm_num [quotient4]

theorem quotient4_exact : quotient4 * prime4 = modulus := by
  norm_num [quotient4, prime4, modulus]

def A : Nat := 819200
theorem A_eval : A = action := by
  norm_num [A, action]

def C : Nat := 409600
theorem C_eval : C = m * n * half := by
  norm_num [C, half, m, n]

def H : Nat := 51200
theorem H_eval : H = m * half := by
  norm_num [H, half, m]

def D : Nat := 10240000
theorem D_eval : D = m * m * n * half := by
  norm_num [D, half, m, n]

def F : Nat := 0
theorem F_eval : F = 0 := by
  norm_num [F]

def members : Nat := 8
theorem members_eval : members = 8 := by
  norm_num [members]

def branches : Nat := 8
theorem branches_eval : branches = 8 := by
  norm_num [branches]

def digits : Nat := 1
theorem digits_eval : digits = 1 := by
  norm_num [digits]

def b1_root : Nat := 1
theorem b1_root_eval : b1_root = 1 := by
  norm_num [b1_root]

def b1_bucket1 : Nat := 1342178918400
theorem b1_bucket1_eval : b1_bucket1 = 2 * (A * (A * b1_root + b1_root) + n * F) := by
  norm_num [b1_bucket1, A, F, b1_root, n]

def b1_bucket2 : Nat := 2199027282083840000
theorem b1_bucket2_eval : b1_bucket2 = 2 * (A * (A * b1_root + b1_bucket1) + n * F) := by
  norm_num [b1_bucket2, A, F, b1_bucket1, b1_root, n]

def b1_bucket : Nat := 2199027282083840000
theorem b1_bucket_eval : b1_bucket = b1_bucket2 := by
  norm_num [b1_bucket, b1_bucket2]

def b1_helper : Nat := 1
theorem b1_helper_eval : b1_helper = max (b1_root) ( F) := by
  norm_num [b1_helper, F, b1_root]

def b1_readout : Nat := 14411545195864653830553600
theorem b1_readout_eval : b1_readout = branches * (A * b1_helper + A * b1_bucket + n * F) := by
  norm_num [b1_readout, A, F, b1_bucket, b1_helper, branches, n]

def b1_sample : Nat := 94447502595618595343916079513600
theorem b1_sample_eval : b1_sample = branches * (A * b1_helper + A * b1_readout + n * F) := by
  norm_num [b1_sample, A, F, b1_helper, b1_readout, branches, n]

def b1_packed : Nat := 755580020764948762751328636108800
theorem b1_packed_eval : b1_packed = members * b1_sample := by
  norm_num [b1_packed, b1_sample, members]

def b1_state : Nat := 755580020764948762751328636108800
theorem b1_state_eval : b1_state = b1_packed := by
  norm_num [b1_state, b1_packed]

def b1_g : Nat := 309485576505323013222944209350164480000
theorem b1_g_eval : b1_g = C * b1_packed := by
  norm_num [b1_g, C, b1_packed]

def b1_one : Nat := 409600
theorem b1_one_eval : b1_one = C * b1_root := by
  norm_num [b1_one, C, b1_root]

def b1_public : Nat := 126765292136580306216117948149827371008000000
theorem b1_public_eval : b1_public = C * b1_g := by
  norm_num [b1_public, C, b1_g]

def b1_labelLevel1 : Nat := 819208
theorem b1_labelLevel1_eval : b1_labelLevel1 = (A + n) * b1_root := by
  norm_num [b1_labelLevel1, A, b1_root, n]

def b1_labelLevel2 : Nat := 671101747264
theorem b1_labelLevel2_eval : b1_labelLevel2 = (A + n) * b1_labelLevel1 := by
  norm_num [b1_labelLevel2, A, b1_labelLevel1, n]

def b1_labelLevel3 : Nat := 549771920172646912
theorem b1_labelLevel3_eval : b1_labelLevel3 = (A + n) * b1_labelLevel2 := by
  norm_num [b1_labelLevel3, A, b1_labelLevel2, n]

def b1_labelProduct : Nat := 549771920172646912
theorem b1_labelProduct_eval : b1_labelProduct = b1_labelLevel3 := by
  norm_num [b1_labelProduct, b1_labelLevel3]

def b1_label : Nat := 1014122337092642634201788694623709655203840000
theorem b1_label_eval : b1_label = 2 * C * C * b1_labelProduct + n * b1_public := by
  norm_num [b1_label, C, b1_labelProduct, b1_public, n]

def b1_automorphism : Nat := 103846127318286586852243823124338719768707072000000
theorem b1_automorphism_eval : b1_automorphism = C * C * (A * b1_helper + A * b1_packed + n * F) := by
  norm_num [b1_automorphism, A, C, F, b1_helper, b1_packed, n]

def b1_private : Nat := 409600
theorem b1_private_eval : b1_private = C * b1_root := by
  norm_num [b1_private, C, b1_root]

def b1_stateTarget : Nat := 309485576505323013222944209350164480000
theorem b1_stateTarget_eval : b1_stateTarget = C * b1_state := by
  norm_num [b1_stateTarget, C, b1_state]

def b1_sigma : Nat := 126765292136580306216117948149827371008000000
theorem b1_sigma_eval : b1_sigma = b1_public := by
  norm_num [b1_sigma, b1_public]

def b1_anchor : Nat := 0
theorem b1_anchor_eval : b1_anchor = 0 := by
  norm_num [b1_anchor]

def b1_fresh : Nat := 3169132303414507655402948703745684275200000000
theorem b1_fresh_eval : b1_fresh = C * D * b1_packed := by
  norm_num [b1_fresh, C, D, b1_packed]

def b1_smudge : Nat := 7737139412633075330573605233754112000000
theorem b1_smudge_eval : b1_smudge = digits * D * b1_packed := by
  norm_num [b1_smudge, D, b1_packed, digits]

def b1_hint : Nat := 9509056992
theorem b1_hint_eval : b1_hint = anchor * n * b1_root * cutoff := by
  norm_num [b1_hint, anchor, b1_root, cutoff, n]

def b1_residue : Nat := 4
theorem b1_residue_eval : b1_residue = 4 := by
  norm_num [b1_residue]

def b1_final : Nat := 8046624989138398343796549443113785536992
theorem b1_final_eval : b1_final = b1_hint + C * b1_state + b1_smudge := by
  norm_num [b1_final, C, b1_hint, b1_smudge, b1_state]

def b1_left_final : Nat := 16093249978276796687593098886227571073992
theorem b1_left_final_eval : b1_left_final = 2 * (b1_final + b1_residue) := by
  norm_num [b1_left_final, b1_final, b1_residue]

theorem b1_final_p0 : b1_left_final < quotient0 := by
  norm_num [b1_left_final, quotient0]

def b1_error_private : Nat := 3295905332690500594694397225500745745373376992
theorem b1_error_private_eval : b1_error_private = b1_hint + C * (b1_g + b1_one + b1_private) + b1_fresh + b1_smudge := by
  norm_num [b1_error_private, C, b1_fresh, b1_g, b1_hint, b1_one, b1_private, b1_smudge]

def b1_left_private : Nat := 6591810665381001189388794451001491490746753992
theorem b1_left_private_eval : b1_left_private = 2 * (b1_error_private + b1_residue) := by
  norm_num [b1_left_private, b1_error_private, b1_residue]

theorem b1_private_p0 : b1_left_private < quotient0 := by
  norm_num [b1_left_private, quotient0]

theorem b1_private_p1 : b1_left_private < quotient1 := by
  norm_num [b1_left_private, quotient1]

theorem b1_private_p2 : b1_left_private < quotient2 := by
  norm_num [b1_left_private, quotient2]

theorem b1_private_p3 : b1_left_private < quotient3 := by
  norm_num [b1_left_private, quotient3]

theorem b1_private_p4 : b1_left_private < quotient4 := by
  norm_num [b1_left_private, quotient4]

def b1_error_stateTarget : Nat := 3422670624827080900910515173650572948609216992
theorem b1_error_stateTarget_eval : b1_error_stateTarget = b1_hint + C * (b1_g + b1_one + b1_stateTarget) + b1_fresh + b1_smudge := by
  norm_num [b1_error_stateTarget, C, b1_fresh, b1_g, b1_hint, b1_one, b1_smudge, b1_stateTarget]

def b1_left_stateTarget : Nat := 6845341249654161801821030347301145897218433992
theorem b1_left_stateTarget_eval : b1_left_stateTarget = 2 * (b1_error_stateTarget + b1_residue) := by
  norm_num [b1_left_stateTarget, b1_error_stateTarget, b1_residue]

theorem b1_stateTarget_p0 : b1_left_stateTarget < quotient0 := by
  norm_num [b1_left_stateTarget, quotient0]

theorem b1_stateTarget_p1 : b1_left_stateTarget < quotient1 := by
  norm_num [b1_left_stateTarget, quotient1]

theorem b1_stateTarget_p2 : b1_left_stateTarget < quotient2 := by
  norm_num [b1_left_stateTarget, quotient2]

theorem b1_stateTarget_p3 : b1_left_stateTarget < quotient3 := by
  norm_num [b1_left_stateTarget, quotient3]

theorem b1_stateTarget_p4 : b1_left_stateTarget < quotient4 := by
  norm_num [b1_left_stateTarget, quotient4]

def b1_error_public : Nat := 51926359564475983926716605959394791910454401216992
theorem b1_error_public_eval : b1_error_public = b1_hint + C * (b1_g + b1_one + b1_public) + b1_fresh + b1_smudge := by
  norm_num [b1_error_public, C, b1_fresh, b1_g, b1_hint, b1_one, b1_public, b1_smudge]

def b1_left_public : Nat := 103852719128951967853433211918789583820908802433992
theorem b1_left_public_eval : b1_left_public = 2 * (b1_error_public + b1_residue) := by
  norm_num [b1_left_public, b1_error_public, b1_residue]

theorem b1_public_p0 : b1_left_public < quotient0 := by
  norm_num [b1_left_public, quotient0]

theorem b1_public_p1 : b1_left_public < quotient1 := by
  norm_num [b1_left_public, quotient1]

theorem b1_public_p2 : b1_left_public < quotient2 := by
  norm_num [b1_left_public, quotient2]

theorem b1_public_p3 : b1_left_public < quotient3 := by
  norm_num [b1_left_public, quotient3]

theorem b1_public_p4 : b1_left_public < quotient4 := by
  norm_num [b1_left_public, quotient4]

def b1_error_label : Nat := 415387805178479113469647343715096975517070465216992
theorem b1_error_label_eval : b1_error_label = b1_hint + C * (b1_g + b1_one + b1_label) + b1_fresh + b1_smudge := by
  norm_num [b1_error_label, C, b1_fresh, b1_g, b1_hint, b1_label, b1_one, b1_smudge]

def b1_left_label : Nat := 830775610356958226939294687430193951034140930433992
theorem b1_left_label_eval : b1_left_label = 2 * (b1_error_label + b1_residue) := by
  norm_num [b1_left_label, b1_error_label, b1_residue]

theorem b1_label_p0 : b1_left_label < quotient0 := by
  norm_num [b1_left_label, quotient0]

theorem b1_label_p1 : b1_left_label < quotient1 := by
  norm_num [b1_left_label, quotient1]

theorem b1_label_p2 : b1_left_label < quotient2 := by
  norm_num [b1_left_label, quotient2]

theorem b1_label_p3 : b1_left_label < quotient3 := by
  norm_num [b1_left_label, quotient3]

theorem b1_label_p4 : b1_left_label < quotient4 := by
  norm_num [b1_left_label, quotient4]

def b1_error_sigma : Nat := 51926359564475983926716605959394791910454401216992
theorem b1_error_sigma_eval : b1_error_sigma = b1_hint + C * (b1_g + b1_one + b1_sigma) + b1_fresh + b1_smudge := by
  norm_num [b1_error_sigma, C, b1_fresh, b1_g, b1_hint, b1_one, b1_sigma, b1_smudge]

def b1_left_sigma : Nat := 103852719128951967853433211918789583820908802433992
theorem b1_left_sigma_eval : b1_left_sigma = 2 * (b1_error_sigma + b1_residue) := by
  norm_num [b1_left_sigma, b1_error_sigma, b1_residue]

theorem b1_sigma_p0 : b1_left_sigma < quotient0 := by
  norm_num [b1_left_sigma, quotient0]

theorem b1_sigma_p1 : b1_left_sigma < quotient1 := by
  norm_num [b1_left_sigma, quotient1]

theorem b1_sigma_p2 : b1_left_sigma < quotient2 := by
  norm_num [b1_left_sigma, quotient2]

theorem b1_sigma_p3 : b1_left_sigma < quotient3 := by
  norm_num [b1_left_sigma, quotient3]

theorem b1_sigma_p4 : b1_left_sigma < quotient4 := by
  norm_num [b1_left_sigma, quotient4]

def b1_error_automorphism : Nat := 42535373752866091307369570546423536842763162268801216992
theorem b1_error_automorphism_eval : b1_error_automorphism = b1_hint + C * (b1_g + b1_one + b1_automorphism) + b1_fresh + b1_smudge := by
  norm_num [b1_error_automorphism, C, b1_automorphism, b1_fresh, b1_g, b1_hint, b1_one, b1_smudge]

def b1_left_automorphism : Nat := 85070747505732182614739141092847073685526324537602433992
theorem b1_left_automorphism_eval : b1_left_automorphism = 2 * (b1_error_automorphism + b1_residue) := by
  norm_num [b1_left_automorphism, b1_error_automorphism, b1_residue]

theorem b1_automorphism_p0 : b1_left_automorphism < quotient0 := by
  norm_num [b1_left_automorphism, quotient0]

theorem b1_automorphism_p1 : b1_left_automorphism < quotient1 := by
  norm_num [b1_left_automorphism, quotient1]

theorem b1_automorphism_p2 : b1_left_automorphism < quotient2 := by
  norm_num [b1_left_automorphism, quotient2]

theorem b1_automorphism_p3 : b1_left_automorphism < quotient3 := by
  norm_num [b1_left_automorphism, quotient3]

theorem b1_automorphism_p4 : b1_left_automorphism < quotient4 := by
  norm_num [b1_left_automorphism, quotient4]

def b1_error_anchor : Nat := 3295905332690500594694397225500745577601216992
theorem b1_error_anchor_eval : b1_error_anchor = b1_hint + C * (b1_g + b1_one + b1_anchor) + b1_fresh + b1_smudge := by
  norm_num [b1_error_anchor, C, b1_anchor, b1_fresh, b1_g, b1_hint, b1_one, b1_smudge]

def b1_left_anchor : Nat := 6591810665381001189388794451001491155202433992
theorem b1_left_anchor_eval : b1_left_anchor = 2 * (b1_error_anchor + b1_residue) := by
  norm_num [b1_left_anchor, b1_error_anchor, b1_residue]

theorem b1_anchor_p0 : b1_left_anchor < quotient0 := by
  norm_num [b1_left_anchor, quotient0]

theorem b1_anchor_p1 : b1_left_anchor < quotient1 := by
  norm_num [b1_left_anchor, quotient1]

theorem b1_anchor_p2 : b1_left_anchor < quotient2 := by
  norm_num [b1_left_anchor, quotient2]

theorem b1_anchor_p3 : b1_left_anchor < quotient3 := by
  norm_num [b1_left_anchor, quotient3]

theorem b1_anchor_p4 : b1_left_anchor < quotient4 := by
  norm_num [b1_left_anchor, quotient4]

def b0_root : Nat := 0
theorem b0_root_eval : b0_root = 0 := by
  norm_num [b0_root]

def b0_bucket1 : Nat := 0
theorem b0_bucket1_eval : b0_bucket1 = 2 * (A * (A * b0_root + b0_root) + n * F) := by
  norm_num [b0_bucket1, A, F, b0_root, n]

def b0_bucket2 : Nat := 0
theorem b0_bucket2_eval : b0_bucket2 = 2 * (A * (A * b0_root + b0_bucket1) + n * F) := by
  norm_num [b0_bucket2, A, F, b0_bucket1, b0_root, n]

def b0_bucket : Nat := 0
theorem b0_bucket_eval : b0_bucket = b0_bucket2 := by
  norm_num [b0_bucket, b0_bucket2]

def b0_helper : Nat := 0
theorem b0_helper_eval : b0_helper = max (b0_root) ( F) := by
  norm_num [b0_helper, F, b0_root]

def b0_readout : Nat := 0
theorem b0_readout_eval : b0_readout = branches * (A * b0_helper + A * b0_bucket + n * F) := by
  norm_num [b0_readout, A, F, b0_bucket, b0_helper, branches, n]

def b0_sample : Nat := 0
theorem b0_sample_eval : b0_sample = branches * (A * b0_helper + A * b0_readout + n * F) := by
  norm_num [b0_sample, A, F, b0_helper, b0_readout, branches, n]

def b0_packed : Nat := 0
theorem b0_packed_eval : b0_packed = members * b0_sample := by
  norm_num [b0_packed, b0_sample, members]

def b0_state : Nat := 0
theorem b0_state_eval : b0_state = b0_packed := by
  norm_num [b0_state, b0_packed]

def b0_g : Nat := 0
theorem b0_g_eval : b0_g = C * b0_packed := by
  norm_num [b0_g, C, b0_packed]

def b0_one : Nat := 0
theorem b0_one_eval : b0_one = C * b0_root := by
  norm_num [b0_one, C, b0_root]

def b0_public : Nat := 0
theorem b0_public_eval : b0_public = C * b0_g := by
  norm_num [b0_public, C, b0_g]

def b0_labelLevel1 : Nat := 0
theorem b0_labelLevel1_eval : b0_labelLevel1 = (A + n) * b0_root := by
  norm_num [b0_labelLevel1, A, b0_root, n]

def b0_labelLevel2 : Nat := 0
theorem b0_labelLevel2_eval : b0_labelLevel2 = (A + n) * b0_labelLevel1 := by
  norm_num [b0_labelLevel2, A, b0_labelLevel1, n]

def b0_labelLevel3 : Nat := 0
theorem b0_labelLevel3_eval : b0_labelLevel3 = (A + n) * b0_labelLevel2 := by
  norm_num [b0_labelLevel3, A, b0_labelLevel2, n]

def b0_labelProduct : Nat := 0
theorem b0_labelProduct_eval : b0_labelProduct = b0_labelLevel3 := by
  norm_num [b0_labelProduct, b0_labelLevel3]

def b0_label : Nat := 0
theorem b0_label_eval : b0_label = 2 * C * C * b0_labelProduct + n * b0_public := by
  norm_num [b0_label, C, b0_labelProduct, b0_public, n]

def b0_automorphism : Nat := 0
theorem b0_automorphism_eval : b0_automorphism = C * C * (A * b0_helper + A * b0_packed + n * F) := by
  norm_num [b0_automorphism, A, C, F, b0_helper, b0_packed, n]

def b0_private : Nat := 0
theorem b0_private_eval : b0_private = C * b0_root := by
  norm_num [b0_private, C, b0_root]

def b0_stateTarget : Nat := 0
theorem b0_stateTarget_eval : b0_stateTarget = C * b0_state := by
  norm_num [b0_stateTarget, C, b0_state]

def b0_sigma : Nat := 0
theorem b0_sigma_eval : b0_sigma = b0_public := by
  norm_num [b0_sigma, b0_public]

def b0_anchor : Nat := 0
theorem b0_anchor_eval : b0_anchor = 0 := by
  norm_num [b0_anchor]

def b0_fresh : Nat := 0
theorem b0_fresh_eval : b0_fresh = C * D * b0_packed := by
  norm_num [b0_fresh, C, D, b0_packed]

def b0_smudge : Nat := 0
theorem b0_smudge_eval : b0_smudge = digits * D * b0_packed := by
  norm_num [b0_smudge, D, b0_packed, digits]

def b0_hint : Nat := 0
theorem b0_hint_eval : b0_hint = anchor * n * b0_root * cutoff := by
  norm_num [b0_hint, anchor, b0_root, cutoff, n]

def b0_residue : Nat := 4
theorem b0_residue_eval : b0_residue = 4 := by
  norm_num [b0_residue]

def b0_final : Nat := 0
theorem b0_final_eval : b0_final = b0_hint + C * b0_state + b0_smudge := by
  norm_num [b0_final, C, b0_hint, b0_smudge, b0_state]

def b0_left_final : Nat := 8
theorem b0_left_final_eval : b0_left_final = 2 * (b0_final + b0_residue) := by
  norm_num [b0_left_final, b0_final, b0_residue]

theorem b0_final_p0 : b0_left_final < quotient0 := by
  norm_num [b0_left_final, quotient0]

def b0_error_private : Nat := 0
theorem b0_error_private_eval : b0_error_private = b0_hint + C * (b0_g + b0_one + b0_private) + b0_fresh + b0_smudge := by
  norm_num [b0_error_private, C, b0_fresh, b0_g, b0_hint, b0_one, b0_private, b0_smudge]

def b0_left_private : Nat := 8
theorem b0_left_private_eval : b0_left_private = 2 * (b0_error_private + b0_residue) := by
  norm_num [b0_left_private, b0_error_private, b0_residue]

theorem b0_private_p0 : b0_left_private < quotient0 := by
  norm_num [b0_left_private, quotient0]

theorem b0_private_p1 : b0_left_private < quotient1 := by
  norm_num [b0_left_private, quotient1]

theorem b0_private_p2 : b0_left_private < quotient2 := by
  norm_num [b0_left_private, quotient2]

theorem b0_private_p3 : b0_left_private < quotient3 := by
  norm_num [b0_left_private, quotient3]

theorem b0_private_p4 : b0_left_private < quotient4 := by
  norm_num [b0_left_private, quotient4]

def b0_error_stateTarget : Nat := 0
theorem b0_error_stateTarget_eval : b0_error_stateTarget = b0_hint + C * (b0_g + b0_one + b0_stateTarget) + b0_fresh + b0_smudge := by
  norm_num [b0_error_stateTarget, C, b0_fresh, b0_g, b0_hint, b0_one, b0_smudge, b0_stateTarget]

def b0_left_stateTarget : Nat := 8
theorem b0_left_stateTarget_eval : b0_left_stateTarget = 2 * (b0_error_stateTarget + b0_residue) := by
  norm_num [b0_left_stateTarget, b0_error_stateTarget, b0_residue]

theorem b0_stateTarget_p0 : b0_left_stateTarget < quotient0 := by
  norm_num [b0_left_stateTarget, quotient0]

theorem b0_stateTarget_p1 : b0_left_stateTarget < quotient1 := by
  norm_num [b0_left_stateTarget, quotient1]

theorem b0_stateTarget_p2 : b0_left_stateTarget < quotient2 := by
  norm_num [b0_left_stateTarget, quotient2]

theorem b0_stateTarget_p3 : b0_left_stateTarget < quotient3 := by
  norm_num [b0_left_stateTarget, quotient3]

theorem b0_stateTarget_p4 : b0_left_stateTarget < quotient4 := by
  norm_num [b0_left_stateTarget, quotient4]

def b0_error_public : Nat := 0
theorem b0_error_public_eval : b0_error_public = b0_hint + C * (b0_g + b0_one + b0_public) + b0_fresh + b0_smudge := by
  norm_num [b0_error_public, C, b0_fresh, b0_g, b0_hint, b0_one, b0_public, b0_smudge]

def b0_left_public : Nat := 8
theorem b0_left_public_eval : b0_left_public = 2 * (b0_error_public + b0_residue) := by
  norm_num [b0_left_public, b0_error_public, b0_residue]

theorem b0_public_p0 : b0_left_public < quotient0 := by
  norm_num [b0_left_public, quotient0]

theorem b0_public_p1 : b0_left_public < quotient1 := by
  norm_num [b0_left_public, quotient1]

theorem b0_public_p2 : b0_left_public < quotient2 := by
  norm_num [b0_left_public, quotient2]

theorem b0_public_p3 : b0_left_public < quotient3 := by
  norm_num [b0_left_public, quotient3]

theorem b0_public_p4 : b0_left_public < quotient4 := by
  norm_num [b0_left_public, quotient4]

def b0_error_label : Nat := 0
theorem b0_error_label_eval : b0_error_label = b0_hint + C * (b0_g + b0_one + b0_label) + b0_fresh + b0_smudge := by
  norm_num [b0_error_label, C, b0_fresh, b0_g, b0_hint, b0_label, b0_one, b0_smudge]

def b0_left_label : Nat := 8
theorem b0_left_label_eval : b0_left_label = 2 * (b0_error_label + b0_residue) := by
  norm_num [b0_left_label, b0_error_label, b0_residue]

theorem b0_label_p0 : b0_left_label < quotient0 := by
  norm_num [b0_left_label, quotient0]

theorem b0_label_p1 : b0_left_label < quotient1 := by
  norm_num [b0_left_label, quotient1]

theorem b0_label_p2 : b0_left_label < quotient2 := by
  norm_num [b0_left_label, quotient2]

theorem b0_label_p3 : b0_left_label < quotient3 := by
  norm_num [b0_left_label, quotient3]

theorem b0_label_p4 : b0_left_label < quotient4 := by
  norm_num [b0_left_label, quotient4]

def b0_error_sigma : Nat := 0
theorem b0_error_sigma_eval : b0_error_sigma = b0_hint + C * (b0_g + b0_one + b0_sigma) + b0_fresh + b0_smudge := by
  norm_num [b0_error_sigma, C, b0_fresh, b0_g, b0_hint, b0_one, b0_sigma, b0_smudge]

def b0_left_sigma : Nat := 8
theorem b0_left_sigma_eval : b0_left_sigma = 2 * (b0_error_sigma + b0_residue) := by
  norm_num [b0_left_sigma, b0_error_sigma, b0_residue]

theorem b0_sigma_p0 : b0_left_sigma < quotient0 := by
  norm_num [b0_left_sigma, quotient0]

theorem b0_sigma_p1 : b0_left_sigma < quotient1 := by
  norm_num [b0_left_sigma, quotient1]

theorem b0_sigma_p2 : b0_left_sigma < quotient2 := by
  norm_num [b0_left_sigma, quotient2]

theorem b0_sigma_p3 : b0_left_sigma < quotient3 := by
  norm_num [b0_left_sigma, quotient3]

theorem b0_sigma_p4 : b0_left_sigma < quotient4 := by
  norm_num [b0_left_sigma, quotient4]

def b0_error_automorphism : Nat := 0
theorem b0_error_automorphism_eval : b0_error_automorphism = b0_hint + C * (b0_g + b0_one + b0_automorphism) + b0_fresh + b0_smudge := by
  norm_num [b0_error_automorphism, C, b0_automorphism, b0_fresh, b0_g, b0_hint, b0_one, b0_smudge]

def b0_left_automorphism : Nat := 8
theorem b0_left_automorphism_eval : b0_left_automorphism = 2 * (b0_error_automorphism + b0_residue) := by
  norm_num [b0_left_automorphism, b0_error_automorphism, b0_residue]

theorem b0_automorphism_p0 : b0_left_automorphism < quotient0 := by
  norm_num [b0_left_automorphism, quotient0]

theorem b0_automorphism_p1 : b0_left_automorphism < quotient1 := by
  norm_num [b0_left_automorphism, quotient1]

theorem b0_automorphism_p2 : b0_left_automorphism < quotient2 := by
  norm_num [b0_left_automorphism, quotient2]

theorem b0_automorphism_p3 : b0_left_automorphism < quotient3 := by
  norm_num [b0_left_automorphism, quotient3]

theorem b0_automorphism_p4 : b0_left_automorphism < quotient4 := by
  norm_num [b0_left_automorphism, quotient4]

def b0_error_anchor : Nat := 0
theorem b0_error_anchor_eval : b0_error_anchor = b0_hint + C * (b0_g + b0_one + b0_anchor) + b0_fresh + b0_smudge := by
  norm_num [b0_error_anchor, C, b0_anchor, b0_fresh, b0_g, b0_hint, b0_one, b0_smudge]

def b0_left_anchor : Nat := 8
theorem b0_left_anchor_eval : b0_left_anchor = 2 * (b0_error_anchor + b0_residue) := by
  norm_num [b0_left_anchor, b0_error_anchor, b0_residue]

theorem b0_anchor_p0 : b0_left_anchor < quotient0 := by
  norm_num [b0_left_anchor, quotient0]

theorem b0_anchor_p1 : b0_left_anchor < quotient1 := by
  norm_num [b0_left_anchor, quotient1]

theorem b0_anchor_p2 : b0_left_anchor < quotient2 := by
  norm_num [b0_left_anchor, quotient2]

theorem b0_anchor_p3 : b0_left_anchor < quotient3 := by
  norm_num [b0_left_anchor, quotient3]

theorem b0_anchor_p4 : b0_left_anchor < quotient4 := by
  norm_num [b0_left_anchor, quotient4]

def b10_root : Nat := 10
theorem b10_root_eval : b10_root = 10 := by
  norm_num [b10_root]

def b10_bucket1 : Nat := 13421789184000
theorem b10_bucket1_eval : b10_bucket1 = 2 * (A * (A * b10_root + b10_root) + n * F) := by
  norm_num [b10_bucket1, A, F, b10_root, n]

def b10_bucket2 : Nat := 21990272820838400000
theorem b10_bucket2_eval : b10_bucket2 = 2 * (A * (A * b10_root + b10_bucket1) + n * F) := by
  norm_num [b10_bucket2, A, F, b10_bucket1, b10_root, n]

def b10_bucket : Nat := 21990272820838400000
theorem b10_bucket_eval : b10_bucket = b10_bucket2 := by
  norm_num [b10_bucket, b10_bucket2]

def b10_helper : Nat := 10
theorem b10_helper_eval : b10_helper = max (b10_root) ( F) := by
  norm_num [b10_helper, F, b10_root]

def b10_readout : Nat := 144115451958646538305536000
theorem b10_readout_eval : b10_readout = branches * (A * b10_helper + A * b10_bucket + n * F) := by
  norm_num [b10_readout, A, F, b10_bucket, b10_helper, branches, n]

def b10_sample : Nat := 944475025956185953439160795136000
theorem b10_sample_eval : b10_sample = branches * (A * b10_helper + A * b10_readout + n * F) := by
  norm_num [b10_sample, A, F, b10_helper, b10_readout, branches, n]

def b10_packed : Nat := 7555800207649487627513286361088000
theorem b10_packed_eval : b10_packed = members * b10_sample := by
  norm_num [b10_packed, b10_sample, members]

def b10_state : Nat := 7555800207649487627513286361088000
theorem b10_state_eval : b10_state = b10_packed := by
  norm_num [b10_state, b10_packed]

def b10_g : Nat := 3094855765053230132229442093501644800000
theorem b10_g_eval : b10_g = C * b10_packed := by
  norm_num [b10_g, C, b10_packed]

def b10_one : Nat := 4096000
theorem b10_one_eval : b10_one = C * b10_root := by
  norm_num [b10_one, C, b10_root]

def b10_public : Nat := 1267652921365803062161179481498273710080000000
theorem b10_public_eval : b10_public = C * b10_g := by
  norm_num [b10_public, C, b10_g]

def b10_labelLevel1 : Nat := 8192080
theorem b10_labelLevel1_eval : b10_labelLevel1 = (A + n) * b10_root := by
  norm_num [b10_labelLevel1, A, b10_root, n]

def b10_labelLevel2 : Nat := 6711017472640
theorem b10_labelLevel2_eval : b10_labelLevel2 = (A + n) * b10_labelLevel1 := by
  norm_num [b10_labelLevel2, A, b10_labelLevel1, n]

def b10_labelLevel3 : Nat := 5497719201726469120
theorem b10_labelLevel3_eval : b10_labelLevel3 = (A + n) * b10_labelLevel2 := by
  norm_num [b10_labelLevel3, A, b10_labelLevel2, n]

def b10_labelProduct : Nat := 5497719201726469120
theorem b10_labelProduct_eval : b10_labelProduct = b10_labelLevel3 := by
  norm_num [b10_labelProduct, b10_labelLevel3]

def b10_label : Nat := 10141223370926426342017886946237096552038400000
theorem b10_label_eval : b10_label = 2 * C * C * b10_labelProduct + n * b10_public := by
  norm_num [b10_label, C, b10_labelProduct, b10_public, n]

def b10_automorphism : Nat := 1038461273182865868522438231243387197687070720000000
theorem b10_automorphism_eval : b10_automorphism = C * C * (A * b10_helper + A * b10_packed + n * F) := by
  norm_num [b10_automorphism, A, C, F, b10_helper, b10_packed, n]

def b10_private : Nat := 4096000
theorem b10_private_eval : b10_private = C * b10_root := by
  norm_num [b10_private, C, b10_root]

def b10_stateTarget : Nat := 3094855765053230132229442093501644800000
theorem b10_stateTarget_eval : b10_stateTarget = C * b10_state := by
  norm_num [b10_stateTarget, C, b10_state]

def b10_sigma : Nat := 1267652921365803062161179481498273710080000000
theorem b10_sigma_eval : b10_sigma = b10_public := by
  norm_num [b10_sigma, b10_public]

def b10_anchor : Nat := 0
theorem b10_anchor_eval : b10_anchor = 0 := by
  norm_num [b10_anchor]

def b10_fresh : Nat := 31691323034145076554029487037456842752000000000
theorem b10_fresh_eval : b10_fresh = C * D * b10_packed := by
  norm_num [b10_fresh, C, D, b10_packed]

def b10_smudge : Nat := 77371394126330753305736052337541120000000
theorem b10_smudge_eval : b10_smudge = digits * D * b10_packed := by
  norm_num [b10_smudge, D, b10_packed, digits]

def b10_hint : Nat := 95090569920
theorem b10_hint_eval : b10_hint = anchor * n * b10_root * cutoff := by
  norm_num [b10_hint, anchor, b10_root, cutoff, n]

def b10_residue : Nat := 4
theorem b10_residue_eval : b10_residue = 4 := by
  norm_num [b10_residue]

def b10_final : Nat := 80466249891383983437965494431137855369920
theorem b10_final_eval : b10_final = b10_hint + C * b10_state + b10_smudge := by
  norm_num [b10_final, C, b10_hint, b10_smudge, b10_state]

def b10_left_final : Nat := 160932499782767966875930988862275710739848
theorem b10_left_final_eval : b10_left_final = 2 * (b10_final + b10_residue) := by
  norm_num [b10_left_final, b10_final, b10_residue]

theorem b10_final_p0 : b10_left_final < quotient0 := by
  norm_num [b10_left_final, quotient0]

def b10_error_private : Nat := 32959053326905005946943972255007457453733769920
theorem b10_error_private_eval : b10_error_private = b10_hint + C * (b10_g + b10_one + b10_private) + b10_fresh + b10_smudge := by
  norm_num [b10_error_private, C, b10_fresh, b10_g, b10_hint, b10_one, b10_private, b10_smudge]

def b10_left_private : Nat := 65918106653810011893887944510014914907467539848
theorem b10_left_private_eval : b10_left_private = 2 * (b10_error_private + b10_residue) := by
  norm_num [b10_left_private, b10_error_private, b10_residue]

theorem b10_private_p0 : b10_left_private < quotient0 := by
  norm_num [b10_left_private, quotient0]

theorem b10_private_p1 : b10_left_private < quotient1 := by
  norm_num [b10_left_private, quotient1]

theorem b10_private_p2 : b10_left_private < quotient2 := by
  norm_num [b10_left_private, quotient2]

theorem b10_private_p3 : b10_left_private < quotient3 := by
  norm_num [b10_left_private, quotient3]

theorem b10_private_p4 : b10_left_private < quotient4 := by
  norm_num [b10_left_private, quotient4]

def b10_error_stateTarget : Nat := 34226706248270809009105151736505729486092169920
theorem b10_error_stateTarget_eval : b10_error_stateTarget = b10_hint + C * (b10_g + b10_one + b10_stateTarget) + b10_fresh + b10_smudge := by
  norm_num [b10_error_stateTarget, C, b10_fresh, b10_g, b10_hint, b10_one, b10_smudge, b10_stateTarget]

def b10_left_stateTarget : Nat := 68453412496541618018210303473011458972184339848
theorem b10_left_stateTarget_eval : b10_left_stateTarget = 2 * (b10_error_stateTarget + b10_residue) := by
  norm_num [b10_left_stateTarget, b10_error_stateTarget, b10_residue]

theorem b10_stateTarget_p0 : b10_left_stateTarget < quotient0 := by
  norm_num [b10_left_stateTarget, quotient0]

theorem b10_stateTarget_p1 : b10_left_stateTarget < quotient1 := by
  norm_num [b10_left_stateTarget, quotient1]

theorem b10_stateTarget_p2 : b10_left_stateTarget < quotient2 := by
  norm_num [b10_left_stateTarget, quotient2]

theorem b10_stateTarget_p3 : b10_left_stateTarget < quotient3 := by
  norm_num [b10_left_stateTarget, quotient3]

theorem b10_stateTarget_p4 : b10_left_stateTarget < quotient4 := by
  norm_num [b10_left_stateTarget, quotient4]

def b10_error_public : Nat := 519263595644759839267166059593947919104544012169920
theorem b10_error_public_eval : b10_error_public = b10_hint + C * (b10_g + b10_one + b10_public) + b10_fresh + b10_smudge := by
  norm_num [b10_error_public, C, b10_fresh, b10_g, b10_hint, b10_one, b10_public, b10_smudge]

def b10_left_public : Nat := 1038527191289519678534332119187895838209088024339848
theorem b10_left_public_eval : b10_left_public = 2 * (b10_error_public + b10_residue) := by
  norm_num [b10_left_public, b10_error_public, b10_residue]

theorem b10_public_p0 : b10_left_public < quotient0 := by
  norm_num [b10_left_public, quotient0]

theorem b10_public_p1 : b10_left_public < quotient1 := by
  norm_num [b10_left_public, quotient1]

theorem b10_public_p2 : b10_left_public < quotient2 := by
  norm_num [b10_left_public, quotient2]

theorem b10_public_p3 : b10_left_public < quotient3 := by
  norm_num [b10_left_public, quotient3]

theorem b10_public_p4 : b10_left_public < quotient4 := by
  norm_num [b10_left_public, quotient4]

def b10_error_label : Nat := 4153878051784791134696473437150969755170704652169920
theorem b10_error_label_eval : b10_error_label = b10_hint + C * (b10_g + b10_one + b10_label) + b10_fresh + b10_smudge := by
  norm_num [b10_error_label, C, b10_fresh, b10_g, b10_hint, b10_label, b10_one, b10_smudge]

def b10_left_label : Nat := 8307756103569582269392946874301939510341409304339848
theorem b10_left_label_eval : b10_left_label = 2 * (b10_error_label + b10_residue) := by
  norm_num [b10_left_label, b10_error_label, b10_residue]

theorem b10_label_p0 : b10_left_label < quotient0 := by
  norm_num [b10_left_label, quotient0]

theorem b10_label_p1 : b10_left_label < quotient1 := by
  norm_num [b10_left_label, quotient1]

theorem b10_label_p2 : b10_left_label < quotient2 := by
  norm_num [b10_left_label, quotient2]

theorem b10_label_p3 : b10_left_label < quotient3 := by
  norm_num [b10_left_label, quotient3]

theorem b10_label_p4 : b10_left_label < quotient4 := by
  norm_num [b10_left_label, quotient4]

def b10_error_sigma : Nat := 519263595644759839267166059593947919104544012169920
theorem b10_error_sigma_eval : b10_error_sigma = b10_hint + C * (b10_g + b10_one + b10_sigma) + b10_fresh + b10_smudge := by
  norm_num [b10_error_sigma, C, b10_fresh, b10_g, b10_hint, b10_one, b10_sigma, b10_smudge]

def b10_left_sigma : Nat := 1038527191289519678534332119187895838209088024339848
theorem b10_left_sigma_eval : b10_left_sigma = 2 * (b10_error_sigma + b10_residue) := by
  norm_num [b10_left_sigma, b10_error_sigma, b10_residue]

theorem b10_sigma_p0 : b10_left_sigma < quotient0 := by
  norm_num [b10_left_sigma, quotient0]

theorem b10_sigma_p1 : b10_left_sigma < quotient1 := by
  norm_num [b10_left_sigma, quotient1]

theorem b10_sigma_p2 : b10_left_sigma < quotient2 := by
  norm_num [b10_left_sigma, quotient2]

theorem b10_sigma_p3 : b10_left_sigma < quotient3 := by
  norm_num [b10_left_sigma, quotient3]

theorem b10_sigma_p4 : b10_left_sigma < quotient4 := by
  norm_num [b10_left_sigma, quotient4]

def b10_error_automorphism : Nat := 425353737528660913073695705464235368427631622688012169920
theorem b10_error_automorphism_eval : b10_error_automorphism = b10_hint + C * (b10_g + b10_one + b10_automorphism) + b10_fresh + b10_smudge := by
  norm_num [b10_error_automorphism, C, b10_automorphism, b10_fresh, b10_g, b10_hint, b10_one, b10_smudge]

def b10_left_automorphism : Nat := 850707475057321826147391410928470736855263245376024339848
theorem b10_left_automorphism_eval : b10_left_automorphism = 2 * (b10_error_automorphism + b10_residue) := by
  norm_num [b10_left_automorphism, b10_error_automorphism, b10_residue]

theorem b10_automorphism_p0 : b10_left_automorphism < quotient0 := by
  norm_num [b10_left_automorphism, quotient0]

theorem b10_automorphism_p1 : b10_left_automorphism < quotient1 := by
  norm_num [b10_left_automorphism, quotient1]

theorem b10_automorphism_p2 : b10_left_automorphism < quotient2 := by
  norm_num [b10_left_automorphism, quotient2]

theorem b10_automorphism_p3 : b10_left_automorphism < quotient3 := by
  norm_num [b10_left_automorphism, quotient3]

theorem b10_automorphism_p4 : b10_left_automorphism < quotient4 := by
  norm_num [b10_left_automorphism, quotient4]

def b10_error_anchor : Nat := 32959053326905005946943972255007455776012169920
theorem b10_error_anchor_eval : b10_error_anchor = b10_hint + C * (b10_g + b10_one + b10_anchor) + b10_fresh + b10_smudge := by
  norm_num [b10_error_anchor, C, b10_anchor, b10_fresh, b10_g, b10_hint, b10_one, b10_smudge]

def b10_left_anchor : Nat := 65918106653810011893887944510014911552024339848
theorem b10_left_anchor_eval : b10_left_anchor = 2 * (b10_error_anchor + b10_residue) := by
  norm_num [b10_left_anchor, b10_error_anchor, b10_residue]

theorem b10_anchor_p0 : b10_left_anchor < quotient0 := by
  norm_num [b10_left_anchor, quotient0]

theorem b10_anchor_p1 : b10_left_anchor < quotient1 := by
  norm_num [b10_left_anchor, quotient1]

theorem b10_anchor_p2 : b10_left_anchor < quotient2 := by
  norm_num [b10_left_anchor, quotient2]

theorem b10_anchor_p3 : b10_left_anchor < quotient3 := by
  norm_num [b10_left_anchor, quotient3]

theorem b10_anchor_p4 : b10_left_anchor < quotient4 := by
  norm_num [b10_left_anchor, quotient4]

end Dio2NoiseCertificate
