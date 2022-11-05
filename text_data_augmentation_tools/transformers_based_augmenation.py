"""Module with the implementation of tools for text augmentation
based on double translate and paraphrasing based on transformer models."""

import os
import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
import transformers
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# Fix all seeds
SEED = 42
transformers.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Set numbers of CPUs for model inference
torch.set_num_threads(5)


class AbstractTransformersAugmenter(ABC):
    """Abstract class for each augmentation."""

    @abstractmethod
    def get_model_and_tokenizer(
            self,
            model_path: str,
            tokenizer_path: str
    ) -> None:
        """Load model and tokenizer from hard to memory.

        @param model_path: multitask-model path
        @param tokenizer_path: multitask-tokenizer path
        """
        pass

    @abstractmethod
    def get_augmenteted_text(self, text: str) -> None:
        """Generate augmented text data.

        @param text: input raw text
        """
        pass


class MultitaskDoubleTranlator(AbstractTransformersAugmenter):
    def __init__(
            self,
            model_path: str,
            tokenizer_path: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Инициализация эклемпляра класса.

        @param model_path: multitask-model path
        @param tokenizer_path: multitask-tokenizer path
        @param device: device for model inference
        """
        self.device = device
        self.multitask_model, self.multitask_tokenizer = \
            self.get_model_and_tokenizer(
                model_path=model_path,
                tokenizer_path=tokenizer_path
            )

    def get_model_and_tokenizer(
            self,
            model_path: str,
            tokenizer_path: str
    ) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
        """Load multitask-model and multitask-tokenizer from hard to memory.

        @param model_path: multitask-model path
        @param tokenizer_path: multitask-tokenizer path
        @return: multitask-model, multitask-tokenizer
        """
        model_from_file = T5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True)
        tokenizer_from_file = T5Tokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        model_from_file.to(self.device)
        model_from_file.eval()
        return model_from_file, tokenizer_from_file

    def multitask_model_generate(self, text: str) -> str:
        """Generate augmented text by multitask-model.

        @param text: input text
        @return: generated text
        """
        inputs = self.multitask_tokenizer(text, return_tensors="pt").to(
            self.multitask_model.device)
        with torch.inference_mode():
            hypotheses = self.multitask_model.generate(**inputs, num_beams=5)
        return self.multitask_tokenizer.decode(
            hypotheses[0], skip_special_tokens=True)

    def get_augmenteted_text(self, text: str) -> str:
        """Generate augmented text data by double translate of multitask-model.

        @param text: raw input text
        @return: augmented text
        """
        forward_translate = self.multitask_model_generate(
            f'translate ru-en | {text}')
        backward_translate = self.multitask_model_generate(
            f'translate en-ru | {forward_translate}')
        return backward_translate.replace(".", "").strip().lower()


class TransformersAugmenterFactory:
    @staticmethod
    def create_augmenter(augmenter: str) -> MultitaskDoubleTranlator:
        """Create augmenter class instance .

        @param augmenter: augmenter type
        @return: augmenter class instance
        """
        try:
            if augmenter == "MultitaskDoubleTranlator":
                return MultitaskDoubleTranlator(
                    model_path="../models/rut5-base-multitask/model/",
                    tokenizer_path="../models/rut5-base-multitask/tokenizer/"
                )
            raise AssertionError("Augmenter type is not valid.")
        except AssertionError as e:
            print(e)
