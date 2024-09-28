# the test file path
test_file="./eval_data/domain/psychology_test.json"
# the test file output file
output_file="./result/psychology_result.txt"
# specify the gpu
cuda_visibale_gpu_index=0

CUDA_VISIBLE_DEVICES=$cuda_visibale_gpu_index python generate.py \
  --dev_file $test_file \
  --dev_batch_size 8 \
  --output_file $output_file \

