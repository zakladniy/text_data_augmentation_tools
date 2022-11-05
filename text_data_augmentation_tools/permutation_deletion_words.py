"""Module with the implementation of tools for text augmentation
by deleting and permutating words in the text."""
import re

import numpy as np


class PermDropWord:
    """Augmentation by deleting and permutating words."""

    @staticmethod
    def drop_random_word(text: str, n_drops: int = 1) -> str:
        """Deletion random words from text.

        @param text: input text
        @param n_drops: number of word for deletion
        @return: new text
        """
        words = text.split()
        len_words = len(words)
        if len_words <= 1 or len_words == n_drops:
            return text
        drop_index = list(np.random.randint(0, len_words, n_drops))
        for index in drop_index:
            words[index] = ""
        string_words = " ".join(words)
        string_words = re.sub("\s\s+", " ", string_words).strip()
        return string_words

    @staticmethod
    def swap_words(text: str) -> str:
        """Permutation of random words in text.

        @param text: входящий текст
        @return: измененный текст
        """
        words = text.split()
        len_words = len(words)
        if len_words <= 1:
            return text
        elif len_words == 2:
            words[0], words[1] = words[1], words[0]
            string_words = " ".join(words)
            string_words = re.sub("\s\s+", " ", string_words).strip()
        else:
            swap_index = list(np.random.randint(0, len_words, 2))
            words[swap_index[0]], words[swap_index[1]] = \
                words[swap_index[1]], words[swap_index[0]]
            string_words = " ".join(words)
            string_words = re.sub("\s\s+", " ", string_words).strip()
        return string_words
