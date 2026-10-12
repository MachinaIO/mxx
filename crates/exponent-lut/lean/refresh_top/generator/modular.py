"""Proofs over the facts modules of an export (`<Module>_Facts.lean`, see
`crates/ir-core/src/lean/facts.rs`), independent of the protocol.

A record scope `S` of a generated module has, in its facts module, wire constants
`S.w_<n>_<p> ctx`, its outputs `S.outs ctx`, and `S.facts ctx h : S.Facts ctx outputs`, whose
fields `F.c<k>.<field>` are the conjuncts of fragment `S.constraints_k`.

- `FactsScope` reads the fields (by fragment) and the wires of one scope.
- `destructuring(text)` reads a proof that destructures hypotheses with `generator/patterns.py`
  (`obtain ⟨w, h⟩ := h`, `unfold S.body at h`, `extract_lets -merge <lets> at h`, then per fragment
  `unfold S.constraints_k at h` / `obtain ⟨<atoms>, h⟩ := h`) and maps each name it binds to the
  facts: a `let` name to its wire constant, an atom name to its field (by position, so a proof
  written against renamed wires still maps).
- `rename(text, names)` replaces those names in proof steps (not the parameter names of named
  arguments `(x := …)`).
"""
import re

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


def balanced_args(text):
    """The top-level space-separated arguments of an application `f a b (c d) …`."""
    args, depth, cur = [], 0, ''
    for ch in text:
        if ch in '([{⟨':
            depth += 1
        elif ch in ')]}⟩':
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur:
                args.append(cur)
            cur = ''
        else:
            cur += ch
    if cur:
        args.append(cur)
    return args


def block(lines, head):
    """The indented lines following the line starting with `head`."""
    i = next(i for i, line in enumerate(lines) if line.startswith(head))
    out, j = [], i + 1
    while j < len(lines) and lines[j].startswith('  '):
        out.append(lines[j])
        j += 1
    return out


class FactsScope:
    """The facts of record scope `scope` (unqualified) of module `module` (also its namespace)."""

    def __init__(self, directory, module, scope):
        self.module, self.scope = module, scope
        self.qual = f'{module}.{scope}'
        text = open(f'{directory}/{module}_Facts.lean').read()
        lines = text.split('\n')
        self.fragments = []
        k = 0
        while True:
            try:
                fields = block(lines, f'structure {scope}.Facts_{k} ')
            except StopIteration:
                break
            self.fragments.append([f.strip().split(' : ')[0] for f in fields])
            k += 1
        self.field_fragment = {f: k for k, fs in enumerate(self.fragments) for f in fs}
        # the body's `let`s in order, as the generated module writes them
        source = open(f'{directory}/{module}.lean').read().split('\n')
        self.lets = [re.match(r'\s*let (\S+)', line).group(1)
                     for line in block(source, f'abbrev {scope}.body ') if re.match(r'\s*let ', line)]
        self.wires = set(re.findall(rf'^def {re.escape(scope)}\.(w_\w+) ', text, re.M))

    def wire(self, name, ctx):
        return f'({self.qual}.{name} {ctx})'

    def field(self, facts, name):
        return f'{facts}.c{self.field_fragment[name]}.{name}'


def destructuring(text, hyp, scope, ctx, facts):
    """The names a `patterns.destruct_root` destructuring of `hyp` binds in `text`, mapped to wire
    constants of `ctx` and fields of `facts` of `FactsScope` `scope`, by position."""
    names = {}
    m = re.search(rf'^  extract_lets -merge (.*) at {hyp}$', text, re.M)
    lets = m.group(1).split()
    assert len(lets) == len(scope.lets), (hyp, len(lets), len(scope.lets))
    for mine, theirs in zip(lets, scope.lets):
        if mine.startswith('_') or theirs.startswith('_'):
            continue
        assert theirs in scope.wires, (hyp, theirs)
        names[mine] = scope.wire(theirs, ctx)
    chunks = re.findall(rf'^  unfold \S+\.constraints_(\d+) at {hyp}\n  obtain ⟨(.*)⟩ := {hyp}$', text, re.M)
    assert len(chunks) == len(scope.fragments), (hyp, len(chunks), len(scope.fragments))
    for k, atoms in chunks:
        atoms = [a.strip() for a in atoms.split(',')]
        if atoms[-1] == hyp:
            atoms = atoms[:-1]
        fields = scope.fragments[int(k)]
        assert len(atoms) == len(fields), (hyp, k, len(atoms), len(fields))
        for mine, theirs in zip(atoms, fields):
            if mine != '-':
                names[mine] = f'{facts}.c{k}.{theirs}'
    return names


def rename(text, names):
    """Replace the identifiers `names` maps, except a parameter name `(x := …)`."""
    def sub(m):
        tok = m.group(0)
        if tok not in names:
            return tok
        start, end = m.start(), m.end()
        if text[max(0, start - 1):start] == '(' and text[end:end + 3] == ' :=':
            return tok
        if start > 0 and text[start - 1] in '.':
            return tok
        return names[tok]
    return IDENT.sub(sub, text)
