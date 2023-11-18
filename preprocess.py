def remove_links(text):
    new_text = []
    for t in text.split(" "):
        # t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if ("http" in t) else t
        new_text.append(t)
    return " ".join(new_text)


def strip_characters(t):
    t = t.replace("\n", " ")
    t = t.replace("\t", " ")
    # t = t.replace('[removed]', " ")
    t = t.replace("  ", " ")
    return t


def clean(t):
    t = t.lower()
    t = t.strip()
    t = strip_characters(t)
    t = remove_links(t)
    return t
