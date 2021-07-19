bs=10240
ac=1
dp=0.0
attdp=0.1
gn=0.5
ws=6000
lr=5

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
       	-world_size 4 \
       	-gpu_ranks 0 1 2 3 \
	-rnn_size 64 \
       	-word_vec_size 64 \
	-transformer_ff 256 \
       	-batch_type tokens \
       	-batch_size $bs \
       	-accum_count $ac \
       	-train_steps 40000 \
       	-max_generator_batches 0 \
       	-normalization tokens \
       	-dropout $dp \
	-attention_dropout $attdp \
       	-max_grad_norm $gn \
       	-optim radam \
       	-encoder_type ltransformer \
       	-decoder_type ltransformer \
        -manifold lorentz \
       	-position_encoding \
       	-param_init 0 \
	-warmup_steps $ws \
       	-learning_rate $lr \
	-weight_decay 0 \
      	-decay_method noam \
       	-label_smoothing 0.1 \
      	-data data/iwslt14/iwslt14.tokenized.de-en/processed.noshare \
       	-layers 6 \
	-heads 4 \
       	-report_every 100 \
        -save_checkpoint_steps 1000 \
	-valid_steps 1000 \
	-master_port 1234 \
	-keep_checkpoint 10 \
	-save_model model/iwslt/64/bs=${bs}_ac=${ac}_dp=${dp}_attdp=${attdp}_gn=${gn}_ws=${ws}_lr=${lr}/model
