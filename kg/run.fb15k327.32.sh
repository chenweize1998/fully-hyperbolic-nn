python3 main.py --dataset FB15k-237\
    --num_epochs 500\
    --batch_size 500\
    --nneg 50\
    --lr 0.003\
    --margin 8.0\
    --max_norm 1.5\
    --max_scale 2.5\
    --max_grad_norm 1.0\
    --dim 32\
    --valid_steps 5 \
    --early_stop 20 \
    --cuda True \
    --optimizer radam
