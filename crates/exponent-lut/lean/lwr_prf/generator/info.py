"""Facts of a generated sparse-LWR claim that the proof generator needs.

`Info(d)` reads the row of parameter directory `d` from `../parameters.csv` and the exported claim in
`d/generated`: the CRT moduli, the gadget widths, the block, cell, member, and LUT counts, the key
layout, the LUT automorphisms, and the hash tags of the selector RHS.
"""
import csv
import os
import re


class Info:
    def __init__(self, d):
        self.d = d
        g = lambda f: open(f'{d}/generated/{f}.lean').read()
        self.online = g('Stage_online')
        self.pre = g('Stage_preprocess')
        self.enc = g('Stage_encrypt')
        pid = os.path.basename(os.path.abspath(d))
        rows = list(csv.DictReader(open(f'{d}/../parameters.csv')))
        self.row = next(r for r in rows if r['parameter_id'] == pid)
        r = self.row
        self.average = r['noise_model'] == 'average'
        self.n = int(r['ring_dimension'])
        self.log_n = self.n.bit_length() - 1
        self.dp = int(r['crt_bits']) // int(r['base_bits'])
        self.base = 1 << int(r['base_bits'])
        self.digit = self.base // 2
        mods = sorted({int(m) for m in re.findall(rf'ExactMatrix (\d+) {self.n} ', self.online + self.pre)})
        self.Q = [None] + mods
        self.L = len(mods)
        self.P = {k: self.Q[k] // self.Q[k - 1] for k in range(2, self.L + 1)}
        assert all(self.Q[k] == self.P[k] * self.Q[k - 1] for k in range(2, self.L + 1))
        self.B = int(r['support_weight']) + 3
        assert self.L - self.B == 2
        q = int(r['input_modulus'])
        self.W = q // 2
        self.log_W = self.W.bit_length() - 1
        self.stride = 2 * self.n // q
        self.pprime = int(r['output_modulus']) // 2
        self.M = int(r['lwr_dimension']) + 1
        self.C = self._cells()
        self.sigmas = [int(x) for x in re.search(
            r'selectInt \(\(Int\.ofNat i_0\)\) \[([^\]]*)\]', self.enc).group(1).replace('(', '').replace(')', '').split(', ')]
        assert len(self.sigmas) == self.W
        self.coords = self._coords()
        self.rhs_prefixes = self._rhs_prefixes()
        self.inverse, self.inverse_k = self._inverse()
        assert self.average, 'the generator proves average-case claims'
        self.cutoff = int(re.search(r'gaussianSampleAt tape \(path \+\+ \[\d+\]\) \(\(\d+/1\)\) \((\d+)\)',
                                    self.enc).group(1))

    def m(self, k):
        """Gadget width of the native gadget of `Q k`."""
        return self.dp * k

    def _cells(self):
        """The cell count: the width of the cell families summed by the block-0 loop."""
        loop = self.online[self.online.index('def sequential_generatedRoot_42 '):]
        return int(re.search(r'\(Fin (\d+) → Mxx\.Primitives\.ExactMatrix \d+ \d+ 1 \d+\) × Unit\) \(outputs',
                             loop).group(1))

    def _coords(self):
        out = []
        for b in range(self.B):
            name = f'parallel_generatedRoot_{23 + 22 * b}'
            i = self.online.index(f'def {name} ')
            j = self.online.index('\ndef ', i + 1)
            lst = re.search(r'selectInt \(\(Int\.ofNat i_0\)\) \[([^\]]*)\]', self.online[i:j]).group(1)
            vals = [int(x) for x in lst.replace('(', '').replace(')', '').split(', ')]
            assert len(vals) == self.C
            out.append(vals)
        return out

    def _rhs_prefixes(self):
        tags = []
        for m in re.finditer(r'hashSample \(hashModel\) \(\[([0-9, ]*)\]\)', self.pre):
            s = bytes(int(x) for x in m.group(1).split(', ')).decode()
            if '/rhs/' in s and s not in [t[1] for t in tags]:
                tags.append((m.group(1), s))
        assert len(tags) == self.B, len(tags)
        tags.sort(key=lambda t: int(re.search(r'RhsId\((\d+)\)', t[1]).group(1)))
        return tags

    def _inverse(self):
        q1 = self.Q[1]
        inv = pow(self.W, -1, q1)
        k = (inv * self.W - 1) // q1
        assert inv * self.W == 1 + q1 * k
        return inv, k


def wrap(items, sep, indent, first=0, width=100):
    """Join `items` with `sep`, breaking lines at `width` columns; continuation lines start with
    `indent` spaces, and the first line already holds `first` columns."""
    lines, cur, col = [], '', first
    for k, x in enumerate(items):
        piece = x + (sep if k < len(items) - 1 else '')
        if cur and col + len(piece.rstrip()) > width:
            lines.append(cur.rstrip())
            cur, col = ' ' * indent, indent
        cur += piece
        col += len(piece)
    lines.append(cur.rstrip() if k < len(items) - 1 else cur)
    return '\n'.join(lines)
