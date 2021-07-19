bs=12000
ac=1
dp=0.1
attdp=0
gn=1
ws=8000
lr=5

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
       	-world_size 4 \
       	-gpu_ranks 0 1 2 3 \
	-rnn_size 256 \
       	-word_vec_size 256 \
	-transformer_ff 1024 \
       	-batch_type tokens \
       	-batch_size $bs \
       	-accum_count $ac \
       	-train_steps 200000 \
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
      	-data ./data/wmt17/processed.noshare \
       	-layers 6 \
	-heads 8 \
       	-report_every 500 \
        -save_checkpoint_steps 5000 \
	-valid_steps 5000 \
	--master_port 1270 \
	-save_model model/wmt17/256/bs=${bs}_ac=${ac}_dp=${dp}_attdp=${attdp}_gn=${gn}_ws=${ws}_lr=${lr}/model
