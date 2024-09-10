import string

punctuation = string.punctuation.replace('_', '')


def split_words(expr: str, reserved: list = None, lower=True):
    new_expr = ''
    expr_len = len(expr)
    skip = 0
    for i in range(expr_len):
        if skip > 0:
            skip -= 1
            continue

        has_reserved = False
        for r in reserved:
            reserve_len = len(r)
            if i <= expr_len - reserve_len and expr[i:i+reserve_len] == r:
                if lower:
                    new_expr += ' ' + r.lower() + ' '
                else:
                    new_expr += ' ' + r + ' '
                skip = reserve_len - 1
                has_reserved = True
                break

        if has_reserved:
            continue
        if expr[i] in punctuation:
            if expr[i] in [',', ';', '.']:
                new_expr += ' '
            else:
                if lower:
                    new_expr += f' {expr[i].lower()} '
                else:
                    new_expr += f' {expr[i]} '
        else:
            if lower:
                new_expr += expr[i].lower()
            else:
                new_expr += expr[i]
    split_words = new_expr.split()
    return split_words


RESERVED = ["&&", "||", "==", '!=']


if __name__ == '__main__':
    words = split_words('return msg.sender ENTRY_POINT && abi.encodePacked()', RESERVED)
    print(words)