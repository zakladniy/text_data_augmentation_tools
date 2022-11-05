"""Module with the implementation of tools for text augmentation
based on double translate and paraphrasing based on transformer models."""

import os
import random
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch
import transformers
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

warnings.simplefilter("ignore", UserWarning)


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
    """Multitask-t5-model augmentator."""
    def __init__(
            self,
            model_path: str,
            tokenizer_path: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialization class instance.

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
            self.device)
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


class HelsinkikDoubleTranlator(AbstractTransformersAugmenter):
    """Helsinki models augmentator."""
    def __init__(
            self,
            helsinki_ru_en_model_path: str,
            helsinki_ru_en_tokenizer_path: str,
            helsinki_en_ru_model_path: str,
            helsinki_en_ru_tokenizer_path: str,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialization class instance.

        @param helsinki_ru_en_model_path: helsinki-ru-en model path
        @param helsinki_ru_en_tokenizer_path: helsinki-ru-en tokenizer path
        @param helsinki_en_ru_model_path: helsinki-en-ru model path
        @param helsinki_en_ru_tokenizer_path: helsinki-en-ru tokenizer path
        @param device: device for model inference
        """
        self.device = device
        self.helsinki_ru_en_model_path, self.helsinki_ru_en_tokenizer_path = \
            self.get_model_and_tokenizer(
                model_path=helsinki_ru_en_model_path,
                tokenizer_path=helsinki_ru_en_tokenizer_path
            )
        self.helsinki_en_ru_model_path, self.helsinki_en_ru_tokenizer_path = \
            self.get_model_and_tokenizer(
                model_path=helsinki_en_ru_model_path,
                tokenizer_path=helsinki_en_ru_tokenizer_path
            )

    def get_model_and_tokenizer(
            self,
            model_path: str,
            tokenizer_path: str
    ) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """Load helsinki-model and helsinki-tokenizer from hard to memory.

        @param model_path: helsinki-model path
        @param tokenizer_path: helsinki-tokenizer path
        @return: helsinki-model, helsinki-tokenizer
        """
        model_from_file = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, local_files_only=True)
        tokenizer_from_file = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        model_from_file.to(self.device)
        model_from_file.eval()
        return model_from_file, tokenizer_from_file

    def helsinki_model_generate(
            self,
            text: str,
            tokenizer: AutoTokenizer,
            model: AutoModelForSeq2SeqLM
    ) -> str:
        """Generate augmented text by helsinki-model.

        @param text: input text
        @param tokenizer: helsinki-tokenizer
        @param model: helsinki-model
        @return: generated text
        """
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device).input_ids
        with torch.inference_mode():
            outputs = model.generate(input_ids=input_ids, num_beams=5)
            results = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)[0]
        return results

    def get_augmenteted_text(self, text: str) -> str:
        """Generate augmented text data by double translate of helsinki-model.

        @param text: raw input text
        @return: augmented text
        """
        forward_translation = self.helsinki_model_generate(
            text=text,
            model=self.helsinki_ru_en_model_path,
            tokenizer=self.helsinki_ru_en_tokenizer_path
        )
        backward_translation = self.helsinki_model_generate(
            text=forward_translation,
            model=self.helsinki_en_ru_model_path,
            tokenizer=self.helsinki_en_ru_tokenizer_path
        )
        return backward_translation.replace(".", "").strip().lower()


class TransformersAugmenterFactory:
    @staticmethod
    def create_augmenter(
            augmenter: str
    ) -> Union[MultitaskDoubleTranlator, HelsinkikDoubleTranlator]:
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
            elif augmenter == "HelsinkikDoubleTranlator":
                return HelsinkikDoubleTranlator(
                    helsinki_ru_en_model_path="../models/helsinki/"
                                              "opus-mt-ru-en/model/",
                    helsinki_ru_en_tokenizer_path="../models/helsinki/"
                                                  "opus-mt-ru-en/tokenizer/",
                    helsinki_en_ru_model_path="../models/helsinki/"
                                              "opus-mt-en-ru/model/",
                    helsinki_en_ru_tokenizer_path="../models/helsinki/"
                                                  "opus-mt-en-ru/tokenizer/"
                )
            else:
                raise AssertionError("Augmenter type is not valid.")
        except AssertionError as e:
            print(e)
