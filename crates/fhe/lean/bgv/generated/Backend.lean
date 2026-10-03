import MxxRuntime

namespace FheBackend

noncomputable section

def moduli0 : List Nat := [1032193]
def layout0 : MxxRuntime.RegularLayout 1032193 :=
  { crtModuli := moduli0
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 1
    base := 2
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 20
    digits_pos := by norm_num
    capacity := by decide }

def moduli1 : List Nat := [18014398507892737]
def layout1 : MxxRuntime.RegularLayout 18014398507892737 :=
  { crtModuli := moduli1
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli2 : List Nat := [18014398508138497]
def layout2 : MxxRuntime.RegularLayout 18014398508138497 :=
  { crtModuli := moduli2
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli3 : List Nat := [18014398508400641]
def layout3 : MxxRuntime.RegularLayout 18014398508400641 :=
  { crtModuli := moduli3
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli4 : List Nat := [18014398507892737, 18014398508138497]
def layout4 : MxxRuntime.RegularLayout 324518553605595287786984016396289 :=
  { crtModuli := moduli4
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli5 : List Nat := [18014398507892737, 72057594037616641]
def layout5 : MxxRuntime.RegularLayout 1298074214513581799797114754236417 :=
  { crtModuli := moduli5
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli6 : List Nat := [18014398507892737, 18014398508138497, 18014398508400641]
def layout6 : MxxRuntime.RegularLayout 5846006548020969210596774788483421161649837621249 :=
  { crtModuli := moduli6
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli7 : List Nat := [18014398507892737, 18014398508138497, 72057594037616641]
def layout7 : MxxRuntime.RegularLayout 23384026193386519304488586269947798086183317045249 :=
  { crtModuli := moduli7
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def moduli8 : List Nat := [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641]
def layout8 : MxxRuntime.RegularLayout 421249166578543632464236954685046510773404157767471124412817604609 :=
  { crtModuli := moduli8
    droppedModuli := 0
    dropped_lt := by decide
    crtModuli_nonempty := by decide
    modulus_pos := by decide
    pairwise_coprime := by unfold Pairwise; decide
    product_eq := by decide
    baseBits := 8
    base := 256
    base_eq := by norm_num
    base_gt_one := by norm_num
    base_even := by decide
    digitsPerTower := 7
    digits_pos := by norm_num
    capacity := by decide }

def backend : MxxRuntime.BackendContext where
  regularLayout q n :=
    if h : q = 1032193 ∧ n = 8192 then
      some (h.1.symm ▸ layout0)
    else
    if h : q = 18014398507892737 ∧ n = 8192 then
      some (h.1.symm ▸ layout1)
    else
    if h : q = 18014398508138497 ∧ n = 8192 then
      some (h.1.symm ▸ layout2)
    else
    if h : q = 18014398508400641 ∧ n = 8192 then
      some (h.1.symm ▸ layout3)
    else
    if h : q = 324518553605595287786984016396289 ∧ n = 8192 then
      some (h.1.symm ▸ layout4)
    else
    if h : q = 1298074214513581799797114754236417 ∧ n = 8192 then
      some (h.1.symm ▸ layout5)
    else
    if h : q = 5846006548020969210596774788483421161649837621249 ∧ n = 8192 then
      some (h.1.symm ▸ layout6)
    else
    if h : q = 23384026193386519304488586269947798086183317045249 ∧ n = 8192 then
      some (h.1.symm ▸ layout7)
    else
    if h : q = 421249166578543632464236954685046510773404157767471124412817604609 ∧ n = 8192 then
      some (h.1.symm ▸ layout8)
    else
    none

end
end FheBackend
