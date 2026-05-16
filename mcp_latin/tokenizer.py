import re
que_exceptions = [] 

# quisque / quique
que_exceptions += [
    "quisque",
    "quidque",
    "quicque",
    "quodque",
    "cuiusque",
    "cuique",
    "quemque",
    "quamque",
    "quoque",
    "quaque",
    "quique",
    "quaeque",
    "quorumque",
    "quarumque",
    "quibusque",
    "quosque",
    "quasque",
]

# uterque
que_exceptions += [
    "uterque",
    "utraque",
    "utrumque",
    "utriusque",
    "utrique",
    "utrumque",
    "utramque",
    "utroque",
    "utraque",
    "utrique",
    "utraeque",
    "utrorumque",
    "utrarumque",
    "utrisque",
    "utrosque",
    "utrasque",
]

# quiscumque
que_exceptions += [
    "quicumque",
    "quidcumque",
    "quodcumque",
    "cuiuscumque",
    "cuicumque",
    "quemcumque",
    "quamcumque",
    "quocumque",
    "quacumque",
    "quicumque",
    "quaecumque",
    "quorumcumque",
    "quarumcumque",
    "quibuscumque",
    "quoscumque",
    "quascumque",
]

# unuscumque
que_exceptions += [
    "unusquisque",
    "unaquaeque",
    "unumquodque",
    "unumquidque",
    "uniuscuiusque",
    "unicuique",
    "unumquemque",
    "unamquamque",
    "unoquoque",
    "unaquaque",
]

# plerusque
que_exceptions += [
    "plerusque",
    "pleraque",
    "plerumque",
    "plerique",
    "pleraeque",
    "pleroque",
    "pleramque",
    "plerorumque",
    "plerarumque",
    "plerisque",
    "plerosque",
    "plerasque",
]

# misc
que_exceptions += [
    "absque",
    "abusque",
    "adaeque",
    "adusque",
    "aeque",
    "antique",
    "atque",
    "circumundique",
    "conseque",
    "cumque",
    "cunque",
    "denique",
    "deque",
    "donique",
    "hodieque",
    "hucusque",
    "inique",
    "inseque",
    "itaque",
    "longinque",
    "namque",
    "neque",
    "oblique",
    "peraeque",
    "praecoque",
    "propinque",
    "qualiscumque",
    "quandocumque",
    "quandoque",
    "quantuluscumque",
    "quantumcumque",
    "quantuscumque",
    "quinque",
    "quocumque",
    "quomodocumque",
    "quomque",
    "quotacumque",
    "quotcumque",
    "quotienscumque",
    "quotiensque",
    "quotusquisque",
    "quousque",
    "relinque",
    "simulatque",
    "torque",
    "ubicumque",
    "ubicunque",
    "ubique",
    "undecumque",
    "undique",
    "usque",
    "usquequaque",
    "utcumque",
    "utercumque",
    "utique",
    "utrimque",
    "utrique",
    "utriusque",
    "utrobique",
    "utrubique",
]


# word/punct tokenizer
#token_pattern = re.compile(r"\w+|[^\w\s]")
token_pattern = re.compile(r"\w+|[^\w\s\[\]<>]")  # exclude [], <>


#sentence_pattern = re.compile(r'(?<=[.!?])\s+')
sentence_pattern = re.compile(r'(?<=[.!])\s+')




def normalize(string: str) -> str:
    """Normalize a text string (e.g. replace v/j with u/i)."""
    trans = str.maketrans({"v": "u", "j": "i", "V": "U", "J": "I"})

    def norm(text: str) -> str:
        return text.translate(trans)

    return norm(string)



def tokenize(text: str) -> list[str]:
    return token_pattern.findall(text)


def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in sentence_pattern.split(text)
        if s.strip()
    ]


def split_que_word(token: str):
    """
    Returns (stem, enclitic) if token should split into X + que/qve,
    otherwise returns None.
    """
    que_exceptions_set = {w.lower() for w in que_exceptions}
    token_norm = token.lower()

    if token_norm in que_exceptions_set:
        return None

    for suffix in ("que", "qve"):
        if token_norm.endswith(suffix) and len(token) > len(suffix):
            stem = token[:-len(suffix)]
            enclitic = token[-len(suffix):]
            return stem, enclitic

    return None


def expand_tokens_with_que(tokens):
    """
    Input: ["auspicia","sortesque","ut","..."]
    Output: a list of dicts describing what to print in CoNLL-U.

    We return a structure like:
    [
        {"type": "single", "form": "auspicia"},
        {"type": "multi", "surface": "sortesque", "parts": ["sortes","que"]},
        {"type": "single", "form": "ut"},
        ...
    ]

    This way we can emit both "2-3 sortesque" and then "2 sortes" / "3 que".
    """
    expanded = []

    for tok in tokens:
        split_result = split_que_word(tok)
        if split_result is None:
            # just a single token
            expanded.append({
                "type": "single",
                "form": tok,
            })
        else:
            stem, enclitic = split_result
            expanded.append({
                "type": "multi",
                "surface": tok,
                "parts": [stem, enclitic],
            })

    return expanded


def post_process_expanded_tokens(expanded):

    token_objs = []

    for item in expanded:

        if item["type"] == "single":
            token = item["form"]

            token_objs.append(token)


                #token_objs.append(
                #        form=item["form"],
                 #       token_type="single"
                 #   )

        else:

            part1, part2 = item["parts"]
            token_objs.append(part1)
            token_objs.append(part2)

    return token_objs



