# Project with tools for text data augmentation

## Description
Contains tools for augmetation russian text data with:
- double translate
  - based on [multitask-t5-model](https://huggingface.co/cointegrated/rut5-base-multitask?text=fill+%7C+%D0%9F%D0%BE%D1%87%D0%B5%D0%BC%D1%83+%D0%BE%D0%BD%D0%B8+%D0%BD%D0%B5+___+%D0%BD%D0%B0+%D0%BC%D0%B5%D0%BD%D1%8F%3F), thanks [David Dale](https://huggingface.co/cointegrated)
  - based on [opus-mt-ru-en](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en?text=%D0%9C%D0%B5%D0%BD%D1%8F+%D0%B7%D0%BE%D0%B2%D1%83%D1%82+%D0%92%D0%BE%D0%BB%D1%8C%D1%84%D0%B3%D0%B0%D0%BD%D0%B3+%D0%B8+%D1%8F+%D0%B6%D0%B8%D0%B2%D1%83+%D0%B2+%D0%91%D0%B5%D1%80%D0%BB%D0%B8%D0%BD%D0%B5) and [opus-mt-en-ru](https://huggingface.co/Helsinki-NLP/opus-mt-en-ru?text=My+name+is+Wolfgang+and+I+live+in+Berlin), thanks [Language Technology Research Group at the University of Helsinki
](https://huggingface.co/Helsinki-NLP)
- paraphrase with [paraphrase-t5-model](https://huggingface.co/cointegrated/rut5-base-paraphraser?text=%D0%9A%D0%B0%D0%B6%D0%B4%D1%8B%D0%B9+%D0%BE%D1%85%D0%BE%D1%82%D0%BD%D0%B8%D0%BA+%D0%B6%D0%B5%D0%BB%D0%B0%D0%B5%D1%82+%D0%B7%D0%BD%D0%B0%D1%82%D1%8C%2C+%D0%B3%D0%B4%D0%B5+%D1%81%D0%B8%D0%B4%D0%B8%D1%82+%D1%84%D0%B0%D0%B7%D0%B0%D0%BD.), thanks [David Dale](https://huggingface.co/cointegrated)
- replacing a word with its synonym #TODO:
  - based on WordNet
  - based on Word2Vec model
- random permutations/deletions of words #TODO
- insertion of parasitic words #TODO
- inserting introductory words #TODO
- inserting typos #TODO

## Project structure
- /models - transformers model, need to download this folder to project directory () 
- /text_data_augmentation_tools - augmentation tools
- /examples - usage examples
- requirements.txt - project Python requirements
- README.md - project description