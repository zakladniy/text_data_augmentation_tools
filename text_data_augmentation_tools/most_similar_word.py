"""Module with the implementation of tools for text augmentation
based on search the most simialr word with WordNet and Word2Vec model."""
from abc import ABC, abstractmethod
from typing import Union

import gensim
import numpy as np
import pymorphy2
from wiki_ru_wordnet import WikiWordnet


class AbstractSimilarWord(ABC):
    """Abstract class for each augmentation."""

    @abstractmethod
    def get_word_normal_form(
            self,
            word: str,
    ) -> None:
        """Get word normal form.

        @param word: input word
        """
        pass

    @abstractmethod
    def get_most_similar_word(self, word: str) -> None:
        """Search the most similar word.

        @param word: input word
        """
        pass


class WordNetSimilarWord(AbstractSimilarWord):
    """Search the most similar word from WordNet."""

    def __init__(self) -> None:
        """Initialization class instance."""
        self.wikiwordnet = WikiWordnet()
        self.morph = pymorphy2.MorphAnalyzer()

    def get_word_normal_form(self, word: str) -> str:
        """Get word normal form.

        @param word: input word
        @return: word normal form
        """
        p = self.morph.parse(word)[0]
        return p.normal_form

    def get_most_similar_word(self, word: str) -> str:
        """Get the most similar word from WordNet.

        @param word: input word
        @return: the most similar word
        """
        normal_word = self.get_word_normal_form(word=word)
        synsets = self.wikiwordnet.get_synsets(normal_word)
        if synsets:
            words = synsets[0].get_words()
            lemmas = [lemma.lemma() for lemma in words]
            random_lemma = np.random.choice(lemmas, size=1)
            return random_lemma[0]
        else:
            return word


class Word2VecSimilarWord(AbstractSimilarWord):
    """Search the most similar word from Word2Vec model."""

    def __init__(self, w2v_model_path: str) -> None:
        """Initialization class instance.

        @param w2v_model_path: word2vec model path
        """
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            w2v_model_path, binary=True)
        self.morph = pymorphy2.MorphAnalyzer()

    def get_word_normal_form(self, word: str) -> str:
        """Get word normal form with POS like "word_POS".

        @param word: input word
        @return: word normal form
        """
        p = self.morph.parse(word)[0]
        normal_form = p.normal_form
        pos = p.tag.POS
        return "_".join([normal_form, pos])

    def get_most_similar_word(self, word: str) -> str:
        """Get the most similar word from Word2Vec.

        @param word: input word
        @return: most similar word
        """
        query = self.get_word_normal_form(word=word)
        try:
            most_similar_words = self.w2v_model.most_similar(
                [query], topn=1)[0][0]
            most_similar_words = most_similar_words.split("_")[0]
        except KeyError:
            most_similar_words = word
        return most_similar_words


class SimilarWordFactory:
    """Create factory with searching most simila word."""

    @staticmethod
    def create_augmenter(
            augmenter: str
    ) -> Union[WordNetSimilarWord, Word2VecSimilarWord]:
        """Create augmenter class instance.

        @param augmenter: augmenter type
        @return: augmenter class instance
        """
        try:
            if augmenter == "WordNetSimilarWord":
                return WordNetSimilarWord()
            elif augmenter == "Word2VecSimilarWord":
                return Word2VecSimilarWord(
                    w2v_model_path="../models/w2v/ruwikiruscorpora_upos_cbow_"
                                   "300_10_2021/model.bin")
            else:
                raise AssertionError("Augmenter type is not valid.")
        except AssertionError as e:
            print(e)
