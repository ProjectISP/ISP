

class Key:
    Ctr = "control"
    Shift = "shift"
    Plus = "+"
    Minus = "-"
    Esc = "escape"
    # TODO include more keys as necessary


# include more key names here if necessary.
POSITIVE_POLARITY_KEYS = [Key.Plus, Key.Shift]

# include more key names here if necessary.
NEGATIVE_POLARITY_KEYS = [Key.Minus, Key.Ctr]


def map_polarity_from_pressed_key(key_name: str):
    polarity = "?"
    color = "red"
    if key_name in POSITIVE_POLARITY_KEYS:
        polarity = "+"
        color = "green"
    elif key_name in NEGATIVE_POLARITY_KEYS:
        polarity = "-"
        color = "blue"
    return polarity, color

