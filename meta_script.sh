#!/bin/bash
hidden_dims=(100 200 500 1000)

for dim in "${hidden_dims[@]}"; do
    echo "Running the script codes/run.py with hidden_dim=${dim}"
    CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
     --cuda \
     --do_valid \
     --do_test \
     --data_path data/FB15k \
     --model RotatE \
     -n 256 -b 1024 -d "$dim" \
     -g 24.0 -a 1.0 -adv \
     -lr 0.0001 --max_steps 150000 \
     -save models/RotatE_FB15k_"$dim" --test_batch_size 16 -de
    if [ $? -ne 0 ]; then 
      echo "Exiting since iteration with dim=${dim} had the error code $?"
      exit 1
    fi
done
