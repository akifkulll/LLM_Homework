"""Rule baseline: lexicon matcher with minimum edit distance (technique area B).

A token is offensive if it matches a lexicon root
  * exactly (all roots), or
  * by prefix (roots >= 4 chars, catches Turkish suffixes: "orospunun", "salaklar"), or
  * within Levenshtein distance 1 of a prefix of the token (roots >= 6 chars; typos: "orosbu",
    "serefsz"), or
  * equal to the root with 1-2 letters deleted (masked / vowel-dropped spellings: "sktr").
"""
from .config import LEXICON_PATH
from .normalize import for_classic


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein; insert/delete/substitute cost 1), dynamic programming."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


class Lexicon:
    def __init__(self, path=LEXICON_PATH):
        self.exact, self.roots, self.phrases, self.allow = set(), [], [], []
        self._cache = {}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                self.allow.append(for_classic(line[1:]))
                continue
            exact_only = line.startswith("=")
            word = for_classic(line.lstrip("="))
            if " " in word:
                self.phrases.append(word)
            elif exact_only or len(word) < 4:
                self.exact.add(word)
            else:
                self.roots.append(word)
        self.roots.sort(key=len, reverse=True)

    def token_hit(self, tok: str):
        """Returns the matched root or None (memoized per token)."""
        if tok not in self._cache:
            self._cache[tok] = self._token_hit(tok)
        return self._cache[tok]

    def _token_hit(self, tok: str):
        if tok in self.exact:
            return tok
        if len(tok) < 3 or not tok.isalpha():
            return None
        if any(tok.startswith(a) for a in self.allow):
            return None
        for r in self.roots:
            if tok.startswith(r):
                return r
        for r in self.roots:
            if len(r) < 6:
                continue
            # typo: a prefix of the token is one edit away from the root ("orosbu", "serefsz")
            for L in (len(r) - 1, len(r), len(r) + 1):
                if L <= len(tok) and edit_distance(tok[:L], r) <= 1:
                    return r
            # masked / vowel-dropped spelling: token is the root with 1-2 letters deleted ("sktr")
            if len(tok) >= 4 and 1 <= len(r) - len(tok) <= 2 and _is_subsequence(tok, r) \
                    and edit_distance(tok, r) == len(r) - len(tok):
                return r
        return None

    def hits(self, text: str):
        norm = for_classic(text)
        toks = norm.split()
        found = [t for t in toks if self.token_hit(t)]
        padded = f" {norm} "
        found += [p for p in self.phrases if f" {p} " in padded]
        return found


class LexiconModel:
    name = "Lexicon + edit distance (rules)"
    short = "lexicon"

    def __init__(self, path=LEXICON_PATH):
        self.lex = Lexicon(path)
        self.path = path
        self.threshold = 0.5

    def predict_proba(self, text: str) -> float:
        n = len(self.lex.hits(text))
        return 0.0 if n == 0 else min(1.0, 0.6 + 0.2 * (n - 1))

    def files(self):
        return [self.path]


if __name__ == "__main__":
    lex = Lexicon()
    for w in ["orospunun", "sktr", "orosbu", "serefsz", "salaklar", "sikayet", "malatya", "amk",
              "aptallık", "adil", "hayvanat", "gotur", "ezikler", "sikinti"]:
        print(f"{w:12} -> {lex.token_hit(w)}")
