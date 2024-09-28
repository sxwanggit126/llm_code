from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='/root/autodl-tmp/code/models', revision='master')
