"""The constants of the decoding chains of refresh_top worst-n8-crt4x30.

Chain `c` decodes at the ring `R` and outputs residues modulo its prime `qc`, `R = Qq qc`. The
action decomposes `X = c' W` at `Mx = R / p`, so `p c' = Qq k` with `k = c' / kdiv`; the online
scale of `G mod qc` is `k3 k1`, `k k3 k1 ≡ q3 (mod qc)`; the mask's sampling table is
`(0, h, 0, h)` with `2 h = qc + 1`.
"""

CHAINS = {
    0: dict(R='Q1', hlayoutR='layoutQ1', qc='q0', Qq=1152919668508929361,
            cp=339699808396096934, kdiv='q1', k3=1073740529, k1=1073741329, h=536870721,
            Mx='M01', hlayoutX='layoutM01', p='q2',
            half=618968813111010066723674601),
    1: dict(R='Q1', hlayoutR='layoutQ1', qc='q1', Qq=1152919788767877569,
            cp=813220753465618156, kdiv='q0', k3=1073740529, k1=112, h=536870665,
            Mx='M01', hlayoutX='layoutM01', p='q2',
            half=618968813111010066723674601),
    2: dict(R='Q1', hlayoutR='layoutQ1', qc='q2', Qq=1152920561861715089,
            cp=885475655076619601, kdiv='q0', k3=1073740529, k1=832, h=536870305,
            Mx='M2', hlayoutX='layoutM2', p='q1',
            half=618968813111010066723674601),
    3: dict(R='D3', hlayoutR='layoutD3', qc='q3', Qq=1152920561861715089,
            cp=585309015821511054, kdiv='q0', k3=0, k1=912, h=536870265,
            Mx='M3', hlayoutX='layoutM3', p='q1',
            half=618968766994187592255071041),
}


# Instance `I` of the merged stages (`i0 < 8`, chain `i0 / 2`): the constants of `TopTables.lean`.
# Integer constants are their tables (as the stages compute with them); natural numbers the moduli.
INSTANCE = dict(R='(R i0)', hlayoutR='(layoutR i0 hi0)', qc='(qc i0)', Qq='(QqZ (i0 : Int))',
                cp='(cpZ (i0 : Int))', kdiv='(kdiv i0)', k3='(k3Z (i0 : Int))', k1='(k1Z (i0 : Int))',
                h='(hmZ (i0 : Int))', Mx='(Mx i0)', hlayoutX='(layoutMx i0 hi0)', p='(pp i0)',
                half='(halfZ (i0 : Int))')


def chain_of(part):
    if part == 'I':
        return INSTANCE
    return CHAINS[int(part.split('_')[0])]


# The substitution of the ring `Q₁` of chains 0–2 by the ring `D3` of chain 3.
D3_SUBST = [
    (r'\blayoutQ1\b', 'layoutD3'),
    (r'\bhalfQ1\b', 'halfD3'), (r'\b618968813111010066723674601\b', '618968766994187592255071041'),
    (r'\b928453219666515100085511901\b', '928453150491281388382606561'),
    (r'\bq0_eq\b', 'q0_eq3'), (r'\bQ1\b', 'D3'),
]


def d3_subst(text, d):
    """`text` at the ring `D3` of parameter directory `d`."""
    import re
    from layouts import layouts
    L = layouts(d)
    text = re.sub(rf'\bBackend\.{L["Q1"]}\b', f'Backend.{L["D3"]}', text)
    for a, b in D3_SUBST:
        text = re.sub(a, b, text)
    return text
