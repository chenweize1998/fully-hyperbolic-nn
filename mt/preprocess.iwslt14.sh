python preprocess.py \
	-train_src data/iwslt14/iwslt14.tokenized.de-en/train.en \
	-train_tgt data/iwslt14/iwslt14.tokenized.de-en/train.de \
	-valid_src data/iwslt14/iwslt14.tokenized.de-en/valid.en \
	-valid_tgt data/iwslt14/iwslt14.tokenized.de-en/valid.de \
	-save_data data/iwslt14/iwslt14.tokenized.de-en/processed.noshare \
	-src_seq_length 256 \
	-tgt_seq_length 256 \
	-src_vocab_size 40000 \
	-tgt_vocab_size 40000 
