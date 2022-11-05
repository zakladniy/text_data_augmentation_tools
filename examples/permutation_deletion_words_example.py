# import time
#
# from text_data_augmentation_tools.permutation_deletion_words import PermDropWord
#
# if __name__ == '__main__':
#
#     text = "добрый день, подскажите пожалуйста как пройти в библиотеку?"
#
#     per_drop_word = PermDropWord()
#
#     # Drop random words
#     start_time = time.time()
#     print(per_drop_word.drop_random_word(text=text, n_drops=1))
#     print(f"Exec time, s: {(time.time() - start_time)}")
#
#     # Swap words
#     start_time = time.time()
#     print(per_drop_word.swap_words(text=text))
#     print(f"Exec time, s: {(time.time() - start_time)}")