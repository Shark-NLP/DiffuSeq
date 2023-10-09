CUDA_VISIBLE_DEVICES=2 python -u run_decode_solver.py \
--model_dir diffusion_models/{name-of-model} \
--seed 110 \
--bsz 100 \
--step 10 \
--split test
