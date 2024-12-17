from math import factorial


def all_pairings(elements):
    if len(elements) == 2:
        return [[(elements[0], elements[1])]]
    else:
        res = []
        first = elements[0]
        for i in range(1, len(elements)):
            pair = (first, elements[i])
            rest = elements[1:i] + elements[i + 1:]
            for sub in all_pairings(rest):
                res.append([pair] + sub)
        return res


def permutation_sign(original, new):
    """
    Compute the sign of the permutation that takes list 'original' to list 'new'.
    We assume 'new' is a rearrangement of 'original'.
    """
    sign = 1
    temp = list(original)
    for i in range(len(temp)):
        if temp[i] != new[i]:
            # Find the index of new[i] in temp
            j = temp.index(new[i], i)
            # Swap elements in temp
            temp[i], temp[j] = temp[j], temp[i]
            sign = -sign
    return sign


def wick_contraction(elements):
    pairings = all_pairings(elements)
    terms = []
    for p in pairings:
        # Extract the order implied by the pairing:
        # We'll create a sequence by taking each pair in the order they appear in the original list.
        # One natural way: reorder the pairs so that the first element of the pairings appear in
        # the same order as they appear in the original.
        # For sign calculation, let's flatten the pairing in the order chosen by lowest original index.

        # Get indices of each element in the original list
        idx_map = {e: i for i, e in enumerate(elements)}
        # Sort pairs by the first element's original position
        p_sorted = sorted(p, key=lambda pair: idx_map[pair[0]])
        # Flatten
        flat_pairs = [x for pair in p_sorted for x in pair]

        # The sign is determined by how flat_pairs differ from the original order
        sign = permutation_sign(elements, flat_pairs)

        # Create the contraction string
        term = "".join(f"<{x}{y}>" for (x, y) in p)
        if sign == -1:
            term = f" -  {term}"
        else:
            term = f" + {term}"
        terms.append(term)
    return " ".join(terms)


# Example
ops = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
wicks = wick_contraction(ops)
print("<"+"".join(ops)+">="+wicks)
print(len(wicks))
