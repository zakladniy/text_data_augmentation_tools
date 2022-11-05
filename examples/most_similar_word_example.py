"""Module with usage examples of augmentation text data
based on searching the most similar word in WordNet and by Word2Vec model."""
import time

from text_data_augmentation_tools.most_similar_word import SimilarWordFactory

if __name__ == '__main__':
    word = "компьютер"

    # MultitaskDoubleTranlator
    augmenter = SimilarWordFactory.create_augmenter("WordNetSimilarWord")
    start_time = time.time()
    new_text_data = augmenter.get_most_similar_word(word=word)
    print(f"Search time, s: {(time.time() - start_time)}")
    print(f"The most similar word: {new_text_data}\n\n")

    # Word2VecSimilarWord
    augmenter = SimilarWordFactory.create_augmenter("Word2VecSimilarWord")
    start_time = time.time()
    new_text_data = augmenter.get_most_similar_word(word=word)
    print(f"Search time, s: {(time.time() - start_time)}")
    print(f"The most similar word: {new_text_data}\n\n")
