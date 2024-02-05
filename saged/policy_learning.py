from difflib import SequenceMatcher
import numpy as np

# TODO documentation
class Transformation():
    def __init__(self, remove="", add=""):
        # Identity transformation
        if remove == add:
            self.remove = ""
            self.add = ""
        else:
            self.remove = remove
            self.add = add

    def apply(self, v):
        position = -1

        if self.remove != "" and self.remove in v:
            # Get all positions of substring self.remove in v
            position_candidates = [
                pos for pos in range(len(v)) if v[pos:pos+len(self.remove)] == self.remove
            ]

            # Remove substring at uniformly random position
            position = np.random.choice(position_candidates)
            v = v[:position] + v[position+len(self.remove):]

        if self.add != "":
            # If no substring was removed, pick a random position for self.add
            if position == -1:
                position = np.random.randint(len(v))

            v = v[:position] + self.add + v[position+1:]

        return v

    def __str__(self):
        if self.remove != "" and self.add != "":
            return self.remove + " -> " + self.add
        if self.remove != "":
            return self.remove + " ↦ ()"
        if self.add != "":
            return "() -> " + self.add

        return "() -> ()"

    def __eq__(self, other):
        return self.remove  == other.remove and self.add == other.add

    def __hash__(self):
        return hash((self.remove, self.add))

def __similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def learn_transformations(v, v_err):
    print(v, v_err)
    if v == "" and v_err == "":
        return []

    transformations = [Transformation(v, v_err)]

    if len(v) <= 1 and len(v_err) <= 1:
        return transformations

    match = SequenceMatcher(None, v, v_err).find_longest_match()

    lv = v[:match.a]
    rv = v[match.a + match.size:]

    lv_err = v_err[:match.b]
    rv_err = v_err[match.b + match.size:]

    if __similarity(lv, lv_err) + __similarity(rv, rv_err) > \
        __similarity(lv, rv_err) + __similarity(rv, lv_err):
        transformations.extend(learn_transformations(lv, lv_err))
        transformations.extend(learn_transformations(rv, rv_err))
    else:
        transformations.extend(learn_transformations(lv, rv_err))
        transformations.extend(learn_transformations(rv, lv_err))

    # Remove identical transformations
    unique = list(set(transformations))

    try:
        # Remove identity transformation () -> ()
        unique.remove(Transformation("", ""))
    except ValueError:
        pass

    return unique
