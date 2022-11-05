"""Module with usage examples of augmentation text data
based on double translate and paraphrasing based on transformer models"""
import time

from text_data_augmentation_tools.transformers_based_augmenation import (
    TransformersAugmenterFactory,
)

if __name__ == '__main__':

    text = "добрый день, сколько сейчас время?"

    # MultitaskDoubleTranlator
    augmenter = TransformersAugmenterFactory.create_augmenter(
        "MultitaskDoubleTranlator")
    start_time = time.time()
    new_text_data = augmenter.get_augmenteted_text(text=text)
    print(f"Generate time, s: {(time.time() - start_time)}")
    print(f"Multitask-model generated text: {new_text_data}\n\n")

    # HelsinkikDoubleTranlator
    augmenter = TransformersAugmenterFactory.create_augmenter(
        "HelsinkikDoubleTranlator")
    start_time = time.time()
    new_text_data = augmenter.get_augmenteted_text(text=text)
    print(f"Generate time, s: {(time.time() - start_time)}")
    print(f"Helsinki-model generated text: {new_text_data}\n\n")
