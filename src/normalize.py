"""Regex normalization of noisy Turkish chat text (technique area B).

Two output flavours:
  for_bert(text)    light clean-up; keeps case and Turkish characters (BERTurk is cased)
  for_classic(text) aggressive: lower-case, Turkish chars folded to ASCII, tokens joined by spaces
                    used by the lexicon matcher, tf-idf and word2vec models
"""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
USER_RE = re.compile(r"@\w+")
# a.m.k  /  s-i-k-t-i-r  /  o*ç  -> letters glued back together
SPACED_LETTERS_RE = re.compile(r"\b(?:[^\W\d_][.\-*_+]){2,}[^\W\d_]\b")
LETTER_SPACE_RE = re.compile(r"(?<!\S)(?:[^\W\d_] ){2,}[^\W\d_](?!\S)")
# 3+ repeats of the same letter -> one  ("amkkkk" -> "amk", "çoook" -> "çok")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")
# masking characters inside a word: "s*kt*r" -> "sktr"
INNER_MASK_RE = re.compile(r"(?<=[^\W\d_])[*_]+(?=[^\W\d_])")
TOKEN_RE = re.compile(r"[a-z0-9]+|[^\w\s.,;:!?'\"()\[\]{}<>/\\|`~^%&=+\-]")

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "€": "e"})
LEET_TOKEN_RE = re.compile(r"\S*[^\W\d_]\S*")
FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _fix_leet(match: re.Match) -> str:
    tok = match.group(0)
    # only touch word cores that mix letters with leet symbols ("s1kt1r", "@pt@l"),
    # not numbers with a suffix ("2020'de") or mentions ("@user")
    core, sep, suffix = tok.partition("'")
    if (re.search(r"[0-9@$€]", core) and re.search(r"[^\W\d_]", core)
            and not core.startswith("@") and not re.fullmatch(r"[\d.,]+[^\W\d_]{1,3}", core)):
        return core.translate(LEET) + sep + suffix
    return tok


def _base(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = SPACED_LETTERS_RE.sub(lambda m: re.sub(r"[.\-*_+]", "", m.group(0)), text)
    text = LETTER_SPACE_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = INNER_MASK_RE.sub("", text)
    text = LEET_TOKEN_RE.sub(_fix_leet, text)
    text = REPEAT_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def for_bert(text: str) -> str:
    text = USER_RE.sub("@user", text)
    return _base(text)


def for_classic(text: str) -> str:
    text = USER_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = tr_lower(_base(text)).translate(FOLD)
    return " ".join(TOKEN_RE.findall(text))


def tokenize(text: str) -> list:
    """Tokens of an already-normalized (for_classic) string."""
    return text.split()


if __name__ == "__main__":
    for s in ["@USER s1kt1r git EZİKKKK", "a.m.k bu ne biçim maç", "o*ç hakem", "ÇOOOOK GÜZEL OLMUŞ 😂😂",
              "s i k t i r", "2020'de 5 gol attı", "https://x.co/a ŞEREFSİZ"]:
        print(f"{s!r:35} -> bert: {for_bert(s)!r:32} classic: {for_classic(s)!r}")
