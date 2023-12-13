# Download NLLB-3.3B model
wget https://pretrained-nmt-models.s3.us-west-2.amazonaws.com/CTranslate2/nllb/nllb-200_3.3B_int8_ct2.zip
unzip nllb-200_3.3B_int8_ct2.zip

# Download SentencePiece model
wget https://pretrained-nmt-models.s3.us-west-2.amazonaws.com/CTranslate2/nllb/flores200_sacrebleu_tokenizer_spm.model

# Download aesthetic scoring model
wget https://nllb-data.com/models/aesthetic-scorer.onnx

# Download NSFW prediction model
wget https://nllb-data.com/models/nsfw-predictor.onnx

# Install dependencies
pip install -r requirements.txt