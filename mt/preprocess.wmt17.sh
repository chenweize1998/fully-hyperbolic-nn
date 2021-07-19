python preprocess.py \
	-train_src data/wmt17/wmt17_en_de/train.en \
	-train_tgt data/wmt17/wmt17_en_de/train.de \
	-valid_src data/wmt17/wmt17_en_de/valid.en \
	-valid_tgt data/wmt17/wmt17_en_de/valid.de \
	-save_data data/wmt17/wmt17_en_de/processed.noshare \
	-src_seq_length 256 \
	-tgt_seq_length 256 \
	-src_vocab_size 50000 \
	-tgt_vocab_size 50000 
