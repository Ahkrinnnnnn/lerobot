# download_model_only_v3.py
#!/usr/bin/env python
"""
专用于下载 LeRobot 预训练模型的脚本 (版本3)。
通过创建最小化的数据集配置来满足 make_policy 的要求。
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Any

import torch

# 尝试导入必要的模块
try:
    from lerobot.policies.factory import make_policy, make_policy_config
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.configs.types import FeatureType
    from lerobot.configs.datasets import BaseDatasetConfig
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_minimal_ds_meta_for_policy(policy_type: str) -> LeRobotDatasetMetadata:
    """
    为指定的策略类型创建一个最小化的数据集元数据（ds_meta）。
    这是一个关键步骤，用于绕过 make_policy 的强制检查。
    注：这里的特征维度是猜测的，对于标准模型（如pi0.5在ALOHA数据集上训练）通常有效。
    """
    # 这是最简化的数据集配置，仅用于提供特征形状信息
    @dataclass
    class MockDatasetConfig(BaseDatasetConfig):
        name: str = "minimal_mock"
        repo_id: str = "lerobot/dummy"
        version: str = "v1"
        # 根据不同的policy_type，我们猜测它可能需要的输入输出特征。
        # pi0/pi0.5 通常处理图像和关节状态，输出动作。
        # 你需要根据你目标任务的实际情况调整这些特征和维度。
        if policy_type in ["pi0", "pi05"]:
            # 假设：1个相机图像（3x224x224），1个关节状态（14维），输出动作（14维）
            input_features = {
                "image": FeatureType.IMAGE,
                "state": FeatureType.STATE,
            }
            output_features = {
                "action": FeatureType.ACTION,
            }
            # 特征形状字典
            features = {
                "image": {"shape": [3, 224, 224], "type": FeatureType.IMAGE},
                "state": {"shape": [14], "type": FeatureType.STATE},
                "action": {"shape": [14], "type": FeatureType.ACTION},
            }
        elif policy_type == "smolvla":
            # 为SmolVLA创建另一个假设配置
            input_features = {
                "image": FeatureType.IMAGE,
                "lang": FeatureType.TEXT,
            }
            output_features = {
                "action": FeatureType.ACTION,
            }
            features = {
                "image": {"shape": [3, 224, 224], "type": FeatureType.IMAGE},
                "lang": {"shape": [], "type": FeatureType.TEXT}, # 文本没有固定形状
                "action": {"shape": [7], "type": FeatureType.ACTION}, # 假设7维动作
            }
        else:
            # 通用回退配置
            input_features = {"observation": FeatureType.STATE}
            output_features = {"action": FeatureType.ACTION}
            features = {
                "observation": {"shape": [10], "type": FeatureType.STATE},
                "action": {"shape": [5], "type": FeatureType.ACTION},
            }

    mock_cfg = MockDatasetConfig()
    
    # 构建一个最小的 LeRobotDatasetMetadata 对象
    # 这是 make_policy 所期望的结构
    ds_meta = LeRobotDatasetMetadata(
        features=mock_cfg.features,
        stats=None,  # 我们没有真实的统计信息，设为None。部分模型可能需要，会报错。
        episodes={
            "dataset_from_index": torch.tensor([0]),
            "dataset_to_index": torch.tensor([1]),
        }
    )
    logger.info(f"为策略 '{policy_type}' 创建了最小化的数据集元数据。")
    logger.info(f"假设特征: {list(mock_cfg.features.keys())}")
    return ds_meta

def download_model(model_id: str, policy_type: str = None):
    """
    通过创建并加载策略来下载模型。
    此版本会创建最小化的 ds_meta 以满足 make_policy 的要求。
    """
    if not IMPORT_SUCCESS:
        logger.error(f"导入 LeRobot 模块失败: {IMPORT_ERROR}")
        logger.error("请确保已正确安装 lerobot 包，并处于正确的开发环境中。")
        return None

    if policy_type is None:
        policy_type = model_id.split('/')[-1].replace('_base', '').replace('_fast', '')

    logger.info(f"开始加载模型: {model_id} (策略类型: '{policy_type}')")
    logger.info("这将触发模型权重的下载（如果尚未缓存）...")

    try:
        # 1. 创建策略配置
        cfg = make_policy_config(policy_type, pretrained_path=model_id, device="cpu")
        logger.info(f"✅ 成功创建配置: {cfg.__class__.__name__}")
        
        # 2. 创建最小化的数据集元数据 (这是关键修复！)
        ds_meta = create_minimal_ds_meta_for_policy(policy_type)
        
        # 3. 现在调用 make_policy，并提供 ds_meta
        logger.info("正在初始化策略模型（这将下载权重）...")
        policy = make_policy(cfg=cfg, ds_meta=ds_meta, env_cfg=None)
        
        logger.info(f"✅ 模型 '{model_id}' 加载（下载）成功！")
        logger.info(f"模型文件已缓存至: ~/.cache/huggingface/hub")
        logger.info(f"模型架构: {policy.__class__.__name__}")
        logger.info("提示：此模型使用虚拟特征维度初始化，若用于实际推理，请提供真实的数据集元数据。")
        return policy
        
    except Exception as e:
        logger.error(f"❌ 过程失败: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # 如果因为特征维度不匹配等原因失败，尝试备选方案
        if "shape" in str(e).lower() or "dimension" in str(e).lower() or "feature" in str(e).lower():
            logger.warning("⚠️  可能虚拟特征维度与实际模型不匹配。")
            logger.warning("   尝试直接下载模型文件...")
            from huggingface_hub import snapshot_download
            local_dir = f"./downloaded_{model_id.replace('/', '_')}"
            try:
                snapshot_download(repo_id=model_id, local_dir=local_dir)
                logger.info(f"✅ 直接下载完成！文件保存在: {local_dir}")
            except ImportError:
                logger.error("需要安装 huggingface_hub: pip install huggingface-hub")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载 LeRobot 预训练模型到本地缓存。')
    parser.add_argument('model_id', type=str, help='Hugging Face 模型ID，例如 lerobot/smolvla_base')
    parser.add_argument('--policy-type', '-t', type=str, default=None,
                        help='(可选) 显式指定策略类型，如 smolvla, pi0, pi05。若不指定，则尝试从model_id推断。')
    args = parser.parse_args()
    
    download_model(args.model_id, args.policy_type)