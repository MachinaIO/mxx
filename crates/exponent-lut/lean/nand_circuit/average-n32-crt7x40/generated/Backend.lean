import MxxRuntime

namespace Backend

noncomputable section

def moduli0 : List Nat := [1099511626049, 1099511625409, 1099511623297]
def layout0 : MxxRuntime.RegularLayout 1329227985420794846003957960673218177 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 20
    base := 1048576
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def moduli1 : List Nat := [1099511626049, 1099511625409, 1099511623297, 1099511622593]
def layout1 : MxxRuntime.RegularLayout 1461501619046042689013583379439969400432371472961 :=
  { crtModuli := moduli1
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 20
    base := 1048576
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def moduli2 : List Nat := [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529]
def layout2 : MxxRuntime.RegularLayout 1606938016486074846153923224048467814823508072370602761938369 :=
  { crtModuli := moduli2
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 20
    base := 1048576
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def moduli3 : List Nat := [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249]
def layout3 : MxxRuntime.RegularLayout 1766847023753256444127058376275403912430991675449840265546213289008802881 :=
  { crtModuli := moduli3
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 20
    base := 1048576
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def moduli4 : List Nat := [1099511626049, 1099511625409, 1099511623297, 1099511622593, 1099511622529, 1099511621249, 1099511619841]
def layout4 : MxxRuntime.RegularLayout 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 :=
  { crtModuli := moduli4
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 20
    base := 1048576
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 2
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 1329227985420794846003957960673218177 ∧ n = 32 then
      some (h.1.symm ▸ layout0)
    else
    if h : q = 1461501619046042689013583379439969400432371472961 ∧ n = 32 then
      some (h.1.symm ▸ layout1)
    else
    if h : q = 1606938016486074846153923224048467814823508072370602761938369 ∧ n = 32 then
      some (h.1.symm ▸ layout2)
    else
    if h : q = 1766847023753256444127058376275403912430991675449840265546213289008802881 ∧ n = 32 then
      some (h.1.symm ▸ layout3)
    else
    if h : q = 1942668833098192796380813666516936640083548573203840422715422556041749136996577561921 ∧ n = 32 then
      some (h.1.symm ▸ layout4)
    else
    none

end
end Backend
