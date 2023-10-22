# NLLB-CLIP

This repository contains the evaluation data for the paper "NLLB-CLIP - train performant multilingual image retrieval model on a budget". To test the performance of the model in 200 languages of the [Flores-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200), I translated the captions for well-known evaluation datasets using [NLLB-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) model.

## XTD200

The files in the `data/xtd200` directory contain the files in the same format as in the original [XTD10 repo](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10). The file name is the Flores-200 language code. To use the dataset, you need to download the [images list](https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/blob/main/XTD10/test_image_names.txt) from the original repo.

## Flickr30k-200

The files in the `data/flickr30k-200` directory contain the captions for the test part of the Flickr30k dataset. The file name is the Flores-200 language code. Original English captions are in the `data/flickr30k-200/eng_captions.txt` file. Image file names are in the `data/flickr30k-200/filenames.txt` file. Images can be found [here](https://huggingface.co/datasets/nlphuji/flickr30k).
