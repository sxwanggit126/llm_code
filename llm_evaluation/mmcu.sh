
# save path
mmcu_save_dir="./result/mmcu/"
# specify cpu index
cuda_visibale_gpu_index=0

CUDA_VISIBLE_DEVICES=$cuda_visibale_gpu_index python mmcu.py \
    --data_dir ./eval_data/MMCU0513 \
    --ntrain 0 \
    --save_dir $mmcu_save_dir \
