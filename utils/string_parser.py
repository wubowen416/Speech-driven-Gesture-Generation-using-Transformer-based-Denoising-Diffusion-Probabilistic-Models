def parse_steps(steps_in_str: str):
    """
    Parse string with "k" at the end to obtain steps in int.
    E.g., "100" -> 100, "500k" -> 500_000, "100kk" -> 100_000_000
    """
    nk = steps_in_str.count("k")
    base = int(steps_in_str.strip("k"))
    if nk == 0:
        steps = base
    else:
        steps = base * nk * 1000
    return steps