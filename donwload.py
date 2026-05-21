from huggingface_hub import snapshot_download

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
# 替换为您的目标文件夹路径，例如 "./my_smolvlm_model"
local_folder_path = "./my_smolvlm_model"

print(f"正在将模型 {model_id} 下载到 {local_folder_path}...")
snapshot_download(repo_id=model_id, local_dir=local_folder_path, local_dir_use_symlinks=False)
print("下载完成！")