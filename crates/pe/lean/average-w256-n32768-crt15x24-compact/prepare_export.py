"""Share the repeated helper-scatter table without changing the exported proposition."""

import re


def normalize_keygen(source):
    if 'def helperScatterIndices : List Int :=' in source:
        return source
    tables = re.findall(r'\[([0-9(), -]+)\]', source)
    table = '[' + max(tables, key=len) + ']'
    if len(re.findall(r'\d+', table)) != 11520 or source.count(table) != 256:
        raise ValueError('Unexpected keygen helper-scatter table shape')
    normalized = source.replace(table, 'helperScatterIndices')
    declaration = f'def helperScatterIndices : List Int := {table}\n\n'
    return normalized.replace('namespace Stage_keygen\n\n',
                              'namespace Stage_keygen\n\n' + declaration, 1)
