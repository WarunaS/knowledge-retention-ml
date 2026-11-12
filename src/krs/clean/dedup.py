"""Near-duplicate removal at conversation level: MinHash LSH (Broder, 1997). Run BEFORE splitting.

Text used for the fingerprint: user and assistant turns, each cut to PER_TURN_CHARS, so a long
shared issue text cannot make two different agent runs on the same task look identical.
One filter instance is shared across all sources, so duplicates between datasets are caught too.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

from krs.schema import Conversation, Role

PER_TURN_CHARS = 300


class NearDuplicateFilter:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128, shingle_words: int = 5, max_chars: int = 5000):
        from datasketch import MinHashLSH

        self.threshold, self.num_perm = threshold, num_perm
        self.shingle_words, self.max_chars = shingle_words, max_chars
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._exact: set[str] = set()

    def _text(self, conv: Conversation) -> str:
        parts = [t.content[:PER_TURN_CHARS] for t in conv.turns if t.role in (Role.USER, Role.ASSISTANT)]
        return " ".join(parts)[: self.max_chars].lower()

    def _shingles(self, text: str) -> set[bytes]:
        words = text.split()
        k = self.shingle_words
        if len(words) < k:
            return {text.encode("utf-8")}
        return {" ".join(words[i : i + k]).encode("utf-8") for i in range(len(words) - k + 1)}

    def is_duplicate(self, conv: Conversation) -> bool:
        """True if a near-duplicate was already seen; otherwise remember this one and return False."""
        from datasketch import MinHash

        text = self._text(conv)
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in self._exact:
            return True
        m = MinHash(num_perm=self.num_perm, seed=1)
        m.update_batch(list(self._shingles(text)))
        if self._lsh.query(m):
            return True
        self._lsh.insert(conv.conversation_id, m, check_duplication=False)
        self._exact.add(digest)
        return False


def deduplicate(convs: Iterable[Conversation], **kwargs) -> Iterator[Conversation]:
    f = NearDuplicateFilter(**kwargs)
    for c in convs:
        if not f.is_duplicate(c):
            yield c
