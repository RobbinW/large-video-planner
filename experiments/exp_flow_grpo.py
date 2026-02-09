import os
import sys
import torch
import math
import logging
import wandb
import tempfile
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from typing import Optional, Dict
import time
import imageio
import json

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from omegaconf import DictConfig, OmegaConf

from .exp_base import BasePytorchExperiment
from algorithms.wan import WanImageToVideo, WanTextToVideo
from algorithms.wan.modules.model import WanAttentionBlock
from datasets.robotwin import RobotwinDataset
from datasets.dummy import DummyVideoDataset
from datasets.mixture import MixtureDataset

# Import Flow-GRPO components
try:
    from flow_grpo.stat_tracking import PerPromptStatTracker
    from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import sde_step_with_logprob
    from flow_grpo.ema import EMAModuleWrapper
except ImportError as e:
    print(f"Failed to import flow_grpo components: {e}")

from flow_grpo.rewards import (
    aesthetic_score, 
    jpeg_compressibility, 
    jpeg_incompressibility,
    clip_score,
    pickscore_score,
    ocr_score,
    video_ocr_score
)

logger = get_logger(__name__)



import contextlib
import time
import numpy as np
import torch
from collections import defaultdict
from concurrent import futures
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.ema import EMAModuleWrapper


# -----------------------------------------------------------------------------
# 1. Sampler (完全对齐)
# -----------------------------------------------------------------------------
class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Per gpu batch size
        self.k = k                    # Repeats per unique prompt (Group Size)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        
        # Calculate distinct samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"Total batch ({self.total_samples}) must be divisible by k ({k})"
        self.m = self.total_samples // self.k  # Number of distinct samples per global batch
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # 1. Randomly select m distinct samples from the dataset
            # If dataset is smaller than needed samples (m), use replacement
            if self.m <= len(self.dataset):
                indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            else:
                indices = torch.randint(0, len(self.dataset), (self.m,), generator=g).tolist()
            
            # 2. Repeat each sample k times
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # 3. Shuffle the repeated list (mix the groups locally and globally)
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # 4. Split for each card
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # 5. Return samples for current rank
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch

# -----------------------------------------------------------------------------
# 2. Utility Functions
# -----------------------------------------------------------------------------
def _to_device(batch, device):
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    return batch


def compute_grpo_loss_per_timestep(new_logp, old_logp, adv, clip_range: float):
    """
    new_logp, old_logp: [B]
    adv: [B]
    """
    ratio = torch.exp(new_logp - old_logp)  # [B]
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.mean(torch.maximum(unclipped, clipped))
    return loss, ratio


# -----------------------------------------------------------------------------
# 3. Trainer Logic (Rollout & Train Step)
# -----------------------------------------------------------------------------
class FlowGRPOTrainer:
    """
    只做算法 loop：采样/奖励/adv/训练
    具体模型推理/条件准备由 wan_algo 提供。
    """

    def __init__(self, wan_algo, reward_fn, cfg, accelerator: Accelerator):
        self.wan_algo = wan_algo
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.accelerator = accelerator

        # Per-prompt tracker
        self.stat_tracker = None
        if cfg.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(cfg.sample.global_std)

    @torch.no_grad()
    def rollout_batch(self, batch, scheduler, timesteps, neg_prompt_embeds=None, determistic: bool = False):
        """
        对齐 flow_grpo: 采样阶段返回
          - videos (用于 reward)
          - latents_traj: [B, S+1, ...]
          - old_log_probs: [B, S]
          - kls: [B, S]（可选）
        """
        device = self.accelerator.device
        B = len(batch["prompt_embeds"])
        # ---- 1) 初始化 latent（I2V 可在这里做你自己的对齐：比如用图片条件影响 init）
        # latents = self.wan_algo.prepare_initial_latents(
        #     batch_size=B,
        #     device=device,
        #     dtype=self.wan_algo._dtype,
        #     batch=batch,  # 给 I2V 用
        # )  # shape: [B, C, T, H, W]

        # clip_embeds = batch["clip_embeds"]
        # image_embeds = batch["image_embeds"]
        # prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]

        batch_size = video_lat.shape[0]

        latents = torch.randn_like(video_lat)

        hist_len =1    # 历史帧长度, 固定1帧
        clean_t = timesteps[-1]  # 最后一个时间步表示 clean

        traj = [latents.clone()]  # all_latents
        old_logps = []  # all_log_probs
        kls = []

        for j, t in enumerate(tqdm(timesteps, desc="Rollout Steps", leave=False, position=1, disable=not self.accelerator.is_local_main_process)):

            # ---- 1) 构造时间步向量
            if self.wan_algo.diffusion_forcing.enabled:
                latents[:, :, :hist_len] = video_lat[:, :, :hist_len]

            t_vec = self.wan_algo.construct_diffusion_forcing_timesteps(
                            batch_size=batch_size,
                            t_curr=t,
                            device=device,
                            clean_t=timesteps[-1],
                            hist_len=hist_len
                        )

            # ---- 2) 采样一步 + logprob（核心对齐）
            # 返回：prev_latents, logp, prev_mean, std_dev_t, dt
            prev_latents, logp, prev_mean, std_dev_t, dt = self.wan_algo.sample_step_with_logprob(
                scheduler=scheduler,
                latents=latents,
                t_vec=t_vec,
                prompt_embeds=batch["prompt_embeds"],
                negative_prompt_embeds=neg_prompt_embeds,
                guidance_scale=self.cfg.sample.guidance_scale,
                do_cfg=self.cfg.train.cfg,
                clip_fea=batch.get("clip_embeds", None),
                y=batch.get("image_embeds", None),
                determistic=determistic,
                debug_label="ROLLOUT-INTERNAL" if (j == 0 and self.accelerator.is_main_process) else None
            )

            traj.append(prev_latents)
            old_logps.append(logp)  # [B]

            # ---- 3) 采样时 KL reward（flow_grpo 的 kl_reward 逻辑可选）
            # 这里简单放 0，占位；如果你要完全一致，可以在 wan_algo 里实现“ref logprob/mean”的那套
            kls.append(torch.zeros_like(logp))

            latents = prev_latents.detach().clone()

        # 堆叠
        latents_traj = torch.stack(traj, dim=1)          # [B, S+1, C, T, H, W]
        old_log_probs = torch.stack(old_logps, dim=1)    # [B, S]
        kl = torch.stack(kls, dim=1)                     # [B, S]

        # decode video
        final_latents = latents_traj[:, -1]
        videos = self.wan_algo.decode_video(final_latents.to(self.wan_algo._dtype))  # [B, C, T, H, W] 或其它

        return videos, latents_traj, old_log_probs, kl

    def compute_advantages(self, prompts_gathered, rewards_gathered):
        """
        对齐 flow_grpo:
        - per_prompt_stat_tracking: 用 tracker 做每个 prompt 的标准化
        - 否则全局标准化
        """
        if self.cfg.per_prompt_stat_tracking:
            advantages = self.stat_tracker.update(prompts_gathered, rewards_gathered)
            group_size, trained_prompt_num = self.stat_tracker.get_stats()
            self.stat_tracker.clear()
            return advantages, group_size, trained_prompt_num
        else:
            # 简单的 Batch 归一化
            # 注意：Flow-GRPO 源码里使用的是 (r - mean) / (std + 1e-4)
            mean = rewards_gathered.mean()
            std = rewards_gathered.std()
            adv = (rewards_gathered - mean) / (std + 1e-4)
            return adv, None, None

    def train_on_samples(self, samples, scheduler, timesteps, optimizer, global_step, ema=None):
            """
            简化版 Flow-GRPO 训练循环
            特点：Tensor 输入，Batch 维度打乱，时间维度顺序训练

            samples: dict of Tensor, 包含
                - latents: [local_B, S, ...]
                - next_latents: [local_B, S, ...]
                - prompt_embeds: [local_B, ...]
                - negative_prompt_embeds: [local_B, ...] (可选)
                - old_log_probs: [local_B, S]
                - advantages: [local_B, S]
            scheduler: diffusion scheduler
            timesteps: Tensor of shape [S], 时间步列表
            """

            accelerator = self.accelerator
            cfg = self.cfg
            model = self.wan_algo.model

            optimizer.zero_grad() # 确保开始前梯度为0

            
            # 1. 获取维度 [B, S, ...]
            total_batch_size = samples["latents"].shape[0]
            print(f"\n[DEBUG]shape of latents: {samples['latents'].shape}, shape of prompt_embeds: {samples['prompt_embeds'].shape}, shape of old_log_probs: {samples['old_log_probs'].shape}, shape of advantages: {samples['advantages'].shape}, shape of timesteps: {timesteps.shape}")
            
            # 确定要训练的时间步 (例如只训练前 50%)
            num_train_timesteps = int(cfg.sample.num_steps * cfg.train.timestep_fraction)
            train_timesteps_val = timesteps[:num_train_timesteps] 
            
            # -------------------------------------------------------
            # 2. Batch Shuffle (优雅且高效的写法)
            # -------------------------------------------------------
            # 生成随机索引 [2, 0, 3, 1...]
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            
            # 对 samples 中所有的 Tensor 应用同一个 perm，保证数据对齐
            # 注意：这里假设 samples 里所有 value 都是 Tensor 且第一维是 Batch
            samples = {k: v[perm] if v is not None else None for k, v in samples.items()}

            # -------------------------------------------------------
            # 3. Micro-Batch 训练循环
            # -------------------------------------------------------
            micro_batch_size = cfg.sample.train_batch_size 
            model.train()
            info_acc = defaultdict(list)

            # --- DEBUG: Check Trainable Params and Gradients ---
            if global_step % 1 == 0 and accelerator.is_main_process:
                 # Check trainable params
                 trainable = [p for p in model.parameters() if p.requires_grad]
                 if len(trainable) == 0:
                     print(f"[DEBUG CRITICAL] Global Step {global_step}: No trainable parameters found! Check LoRA/Freeze logic.")
                 else:
                     # Check if they really have gradients attached (after backward)
                     print(f"[DEBUG] Global Step {global_step}: {len(trainable)} trainable parameters.")
            # ---------------------------------------------------
            
            # 外层：遍历切分后的 Micro-Batch
            micro_batch_iter = range(0, total_batch_size, micro_batch_size)
            print(f"\n[DEBUG] Total Batch Size: {total_batch_size}, Micro Batch Size: {micro_batch_size}, Iterations: {len(micro_batch_iter)}")
            if accelerator.is_local_main_process:
                micro_batch_iter = tqdm(micro_batch_iter, desc="Micro-Batch", leave=False)
            
            for i in micro_batch_iter:
                # 构造切片，例如 slice(0, 2)
                sl = slice(i, min(i + micro_batch_size, total_batch_size))
                current_bs = samples["latents"][sl].shape[0] # 当前 micro batch 实际大小
                
                # 准备当前 Batch 的条件 (Prompt/Negative/Clip)
                # 因为你已经支持 Tensor 输入，直接切片即可！非常清爽
                b_p_embeds = samples["prompt_embeds"][sl]
                b_n_embeds = samples["negative_prompt_embeds"][sl] if samples.get("negative_prompt_embeds") is not None else None
                b_clip = samples["clip_fea"][sl] if samples.get("clip_fea") is not None else None
                b_y = samples["y"][sl] if samples.get("y") is not None else None
                
                # gt_latents 形状通常是 [B, C, F, H, W]，它不随时间步 t 变化
                b_gt = samples["gt_latents"][sl].to(accelerator.device)
                hist_len = 1
                
                # 为了实现 "Per Trajectory" 的 Update，我们在进入时间步循环前清空梯度
                # 但由于我们这里是 Micro-Batch 循环，如果我们需要 Accumulate 多个 Micro-Batch，逻辑会更复杂
                # 现在的逻辑简化为：每个 Micro-Batch (即每个视频) 进行一次 Optimizer Step
                

                # 内层：遍历时间步 (顺序 t=0, t=1...)
                timesteps_iter = enumerate(train_timesteps_val)
                if accelerator.is_local_main_process:
                    timesteps_iter = tqdm(timesteps_iter, desc="Timestep", total=len(train_timesteps_val), leave=False)
                
                for t_idx, t_val in timesteps_iter:
                    
                    # A. 准备当前 step 的数据
                    # latents 形状是 [Batch, Time, ...] -> 切片变成 [Micro_Batch, ...]
                    b_lat = samples["latents"][sl, t_idx]       
                    b_next = samples["next_latents"][sl, t_idx] 

                    b_logp_old = samples["old_log_probs"][sl, t_idx]
                    b_adv = samples["advantages"][sl, t_idx]

                    # 🕵️‍♂️ DEBUG 1: 检查 Advantage 分布
                    if i == 0 and t_idx == 0 and global_step % 1 == 0 and accelerator.is_main_process:
                        print(f"\n[DEBUG Step {global_step}] Advantage Analysis:")
                        print(f"  > Raw Adv Mean: {b_adv.mean().item():.6f}")
                        print(f"  > Raw Adv Max:  {b_adv.max().item():.6f}")
                        print(f"  > Raw Adv Min:  {b_adv.min().item():.6f}")
                        if b_adv.abs().sum().item() < 1e-6:
                             print(f"  ⚠️ CRITICAL: Advantage is ZERO on Main Process. Loss will be zero.")

                    # B. 构造时间步向量
                    if self.wan_algo.diffusion_forcing.enabled:
                        b_lat[:, :, :hist_len] = b_gt[:, :, :hist_len]  # 覆盖历史帧部分

                    b_t_vec = self.wan_algo.construct_diffusion_forcing_timesteps(
                        batch_size=current_bs,
                        t_curr=t_val,  # 当前训练的时间步
                        device=accelerator.device,
                        clean_t=timesteps[-1],
                        hist_len=hist_len
                        ) 
                    
                    # 最后再开启梯度！(Prime the Pump)
                    # 这样 b_lat 带着修改后的值，成为了计算图的起点
                    b_lat.requires_grad_(True)
                    
                    with accelerator.accumulate(model):
                        # C. 计算新 LogProb
                        # 注意：移除 accelerator.accumulate 上下文，手动累积
                        _, new_logp, new_mean, new_std, new_dt = self.wan_algo.logprob_of_transition(
                            scheduler=scheduler,
                            latents=b_lat,
                            next_latents=b_next,
                            t_vec=b_t_vec,
                            prompt_embeds=b_p_embeds,          # 直接传 Tensor
                            negative_prompt_embeds=b_n_embeds, # 直接传 Tensor
                            guidance_scale=cfg.sample.guidance_scale,
                            do_cfg=cfg.train.cfg,
                            clip_fea=b_clip,
                            y=b_y,
                            debug_label="TRAIN-INTERNAL" if (i == 0 and t_idx == 0 and global_step == 0 and accelerator.is_main_process) else None
                        )
                        
                        # --- DEBUG: LogProb divergence ---
                        if i == 0 and t_idx == 0 and accelerator.is_main_process and global_step % 1 == 0:
                            diff = (new_logp.detach() - b_logp_old).abs().mean().item()
                            print(f"[DEBUG] Step {global_step} t={t_idx}: |NewLogP - OldLogP| = {diff:.6f}")
                            if diff < 1e-6:
                                print(f"[DEBUG INFO] Logprobs are identical. This verifies that Training Forward Pass matches Rollout. (Expected for Step 0)")
                        # ---------------------------------

                        # D. GRPO Loss
                        # Clip advantage
                        b_adv_clipped = torch.clamp(b_adv, -cfg.train.adv_clip_max, cfg.train.adv_clip_max)
                        
                        # Ratio = exp(new - old)
                        policy_loss, ratio = compute_grpo_loss_per_timestep(
                            new_logp, b_logp_old, b_adv_clipped, clip_range=cfg.train.clip_range
                        )
                        
                        # --- DEBUG: Verify Computational Graph ---
                        if i == 0 and t_idx == 0:
                            # 检查梯度连接
                            if not new_logp.requires_grad:
                                print(f"[DEBUG CRITICAL] new_logp does not require grad! This means computation graph is broken.")
                            else:
                                # 尝试对一个标量 Backward 看看哪些参数有 grad
                                pass 
                        # ------------------------------------------

                        # E. KL Regularization
              
                        loss = policy_loss
                        kl_loss_val = torch.tensor(0.0, device=accelerator.device)

                        if cfg.train.beta > 0:
                            with torch.no_grad():
                                with model.disable_adapter():
                                    _, _, ref_mean, ref_std, ref_dt = self.wan_algo.logprob_of_transition(
                                        scheduler=scheduler,
                                        latents=b_lat,
                                        next_latents=b_next,
                                        t_vec=b_t_vec,
                                        prompt_embeds=b_p_embeds,
                                        negative_prompt_embeds=b_n_embeds,
                                        guidance_scale=cfg.sample.guidance_scale,
                                        do_cfg=cfg.train.cfg,
                                        clip_fea=b_clip,
                                        y=b_y
                                    )
                            
                            denom = (new_std * ref_dt).pow(2) * 2.0
                            kl_term = ((new_mean - ref_mean).pow(2)).mean() / (denom.mean() + 1e-12)
                            
                            loss = policy_loss + cfg.train.beta * kl_term
                            kl_loss_val = kl_term

                        # 🕵️‍♂️ DEBUG 3: 检查 Loss 成分
                        if i == 0 and t_idx == 0 and global_step % 1 == 0 and accelerator.is_main_process:
                            print(f"  > Ratio Mean:   {ratio.mean().item():.6f}")
                            print(f"  > Policy Loss:  {policy_loss.item():.8f}")
                            print(f"  >  Loss:        {loss.item():.8f}")

                            if cfg.train.beta > 0:
                                print(f"  > KL Term:      {kl_term.item():.8f}")
                                print(f"  > Total Loss:   {loss.item():.8f}")


                        # F. Backward
                        accelerator.backward(loss)
                        
                        if accelerator.sync_gradients:
                            total_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                        
                        # --- DEBUG: Check Gradient Norm Before Step ---
                        if i == 0 and accelerator.is_main_process and global_step % 1 == 0 and accelerator.sync_gradients:
                            # Note: Accelrator handles FSDP grads hidden in .flat_param usually, or specialized access
                            # We assume unwrap or direct access might not show full picture in FSDP sharded, but we can try logging total norm
                            print(f"[DEBUG] Step {global_step}: Total Grad Norm computed by clip_grad_norm: {total_norm.item() if torch.is_tensor(total_norm) else total_norm}")
                        # ---------------------------------------------

                        
                        # 4. Sync 阶段的操作
                        if accelerator.sync_gradients:
                            
                            # [DEBUG STEP 1] Update 前：抓取一个可训练参数做快照
                            # 我们只抓取 LoRA 的参数，因为 Base Model 是冻结的
                            param_to_watch = None
                            param_name_to_watch = ""
                            weight_before = None
                            
                            print(f"\n[DEBUG] Step {global_step}: inner loop iter={i}, timestep={t_idx} before optimizer step. Sync Gradients: {accelerator.sync_gradients}")
                            if accelerator.is_main_process:
                                for name, p in model.named_parameters():
                                    # 找一个有梯度的 LoRA 参数 (最好是 lora_B，因为它初始化为 0，变没变很明显)
                                    if p.requires_grad and p.grad is not None and "lora_B" in name:
                                        # FSDP 下，我们只能看到本地分片，如果有数据就用，没数据就换下一个
                                        if p.numel() > 0: 
                                            param_to_watch = p
                                            param_name_to_watch = name
                                            weight_before = p.detach().clone() # 必须 Clone！否则是指针引用
                                            break
                            
                            # Clip Gradient
                            total_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                            
                            # Debug Print Grad
                            if i == 0 and accelerator.is_main_process:
                                print(f"[DEBUG] Step {global_step}: Norm={total_norm.item():.6f}, Loss={loss.item():.6f}")
                                current_lr = optimizer.param_groups[0]['lr']
                                print(f"[DEBUG CHECK] Current LR: {current_lr:.10f}")
                                if current_lr == 0:
                                    print("💀 DEAD: LR is ZERO!")    

                        # Only zero gradients when we actually synced/stepped
                        # This enables proper gradient accumulation
                        if accelerator.sync_gradients:

                            # 为了防止 Log 太多炸掉，每张卡只打印自己找到的前 3 个非 None 参数
                            my_count = 0
                            rank = accelerator.process_index
                            
                            for name, param in model.named_parameters():
                                if "lora" in name and param.requires_grad:
                                    if param.grad is not None:
                                        grad_norm = param.grad.norm().item()
                                        # 只要不为 None，就说明这张卡负责这部分参数
                                        if grad_norm == 0:
                                            print(f"[Rank {rank}] ⚠️ {name}: Grad is 0.0 (Connected!)")
                                        else:
                                            print(f"[Rank {rank}] ✅ {name}: Grad Norm = {grad_norm:.6f}")
                                        
                                        my_count += 1
                                        if my_count >= 3: break # 每张卡只报 3 个，避免刷屏

                            # 配合 barrier 稍微对齐一下输出时间，虽然后台打印可能还是乱的
                            accelerator.wait_for_everyone()
                            # =======================================================
                            
                        optimizer.step()
                        optimizer.zero_grad()       

                        # 7. Post-Step Logic (Log & EMA)
                        # 只有在真正发生 Step 更新后才执行
                        if accelerator.sync_gradients:
                            # [DEBUG STEP 2] Update 后：对比变化
                            if accelerator.is_main_process and weight_before is not None:
                                weight_after = param_to_watch.detach()
                                
                                # 计算差值 (L1 距离)
                                diff = (weight_after - weight_before).abs().sum().item()
                                max_val = weight_after.abs().max().item()
                                print("\n" + "-"*50)

                                print(f"\n[DEBUG WEIGHT UPDATE] Layer: {param_name_to_watch}")
                                print(f"  > Pre-Update Sum:  {weight_before.abs().sum().item():.8f}")
                                print(f"  > Post-Update Sum: {weight_after.abs().sum().item():.8f}")
                                print(f"  > Total Diff (L1): {diff:.9f}")
                                print(f"  > Max Value:       {max_val:.9f}")
                                
                                if diff == 0.0:
                                    print("  ❌ ALERT: Weights did NOT move at all! (Precision Issue or Zero Grad)")
                                else:
                                    print("  ✅ SUCCESS: Weights moved! Model is updating.")
                                print("-" * 50 + "\n")

                                for n, p in model.named_parameters():
                                    if "lora_B" in n and p.requires_grad:
                                        # 检查 LoRA B 的 L1 范数
                                        weight_sum = p.abs().sum().item()
                                        print(f"[DEBUG CHECK] Step {global_step} {n} L1 Sum: {weight_sum:.9f}")
                                        if weight_sum == 0.0:
                                            print("  💀 DEAD: Weight is still pure ZERO.")
                                        else:
                                            print("  ❤️ ALIVE: Weight is non-zero! Model is learning.")
                                        break # 看一个就够了


                            # Logging
                            info_acc["policy_loss"].append(policy_loss.detach())
                            info_acc["loss"].append(loss.detach())
                            # info_acc["kl_loss"].append(kl_term.detach())
                            info_acc["clipfrac"].append((torch.abs(ratio.detach() - 1.0) > cfg.train.clip_range).float().mean())
                            info_acc["approx_kl"].append(0.5 * ((new_logp.detach() - b_logp_old) ** 2).mean())

                            # EMA Update
                            if cfg.train.ema and ema is not None and accelerator.is_main_process:
                                ema.step(list(filter(lambda p: p.requires_grad, model.parameters())), global_step)

            return {k: torch.stack(v).mean() for k, v in info_acc.items()}



# def compute_log_prob(wan_algo, scheduler, latents, next_latents, timesteps, prompt_embeds, negative_prompt_embeds, guidance_scale, clip_fea=None, y=None, do_classifier_free_guidance=True):
#     """
#     Computes log probability of the transition from latents to next_latents.
#     Adapted for Wan2.1 with clip_fea and y support.
#     """
#     transformer=wan_algo.model
#     if do_classifier_free_guidance:
#         # Predict noise for text
#         noise_pred_text = transformer(
#             latents,
#             t=timesteps,
#             context=prompt_embeds,
#             seq_len=wan_algo.max_tokens,
#             clip_fea=clip_fea,
#             y=y,
#         )[0]
        
#         # Predict noise for uncond
#         noise_pred_uncond = transformer(
#             latents,
#             t=timesteps,
#             context=negative_prompt_embeds,
#             seq_len=wan_algo.max_tokens,
#             clip_fea=clip_fea,
#             y=y,
#         )[0]
        
#         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
#     else:
#         noise_pred = transformer(
#             latents,
#             t=timesteps,
#             context=prompt_embeds,
#             seq_len=wan_algo.max_tokens,
#             clip_fea=clip_fea,
#             y=y,
#         )[0]

#     # Compute log prob using SDE step logic
#     # Assumes sde_step_with_logprob returns: prev_sample, log_prob, prev_sample_mean, std_dev_t, dt
#     prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
#         scheduler,
#         noise_pred.float(),
#         timesteps,
#         latents.float(),
#         prev_sample=next_latents.float(),
#         return_dt_and_std_dev_t=True
#     )
    
#     return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt


class FlowGRPOExperiment(BasePytorchExperiment):
    """
    Experiment for Flow-GRPO Reinforcement Learning.
    Inherits from BasePytorchExperiment but implements a custom training loop with Accelerate
    instead of relying on standard Lightning Trainer, as GRPO requires complex sampling/reward interaction.
    """
    
    compatible_algorithms = dict(
        wan_i2v=WanImageToVideo,
        wan_t2v=WanTextToVideo,
    )
    
    compatible_datasets = dict(
        robotwin=RobotwinDataset,
        dummy=DummyVideoDataset,
        mixture=MixtureDataset,
    )


    def _videos_to_uint8(self, videos: torch.Tensor) -> np.ndarray:
        # videos: [B, C, T, H, W] -> uint8 [B, T, H, W, C]
        videos = videos.detach().float().cpu()
        if videos.min() < 0:
            videos = (videos + 1.0) / 2.0
        videos = videos.clamp(0, 1)
        videos = (videos * 255).round().to(torch.uint8)
        return videos.permute(0, 2, 3, 4, 1).numpy()

    @torch.no_grad()
    def _run_eval(
        self,
        trainer,
        wan_algo,
        eval_loader,
        reward_fn,
        cfg,
        accelerator: Accelerator,
        global_step: int,
        eval_num_batches: int,
        eval_num_videos: int,
        eval_fps: int,
        epoch: int = 0,
    ):
        if eval_loader is None:
            return

        wan_algo.model.eval()
        all_rewards = defaultdict(list)
        logged_videos = None
        logged_prompts = None
        logged_rewards = None

        eval_neg_prompt_embeds = None
        if cfg.train.cfg:
            eval_bs = cfg.sample.get("eval_batch_size", cfg.sample.sample_batch_size)
            eval_neg_prompt_embeds = wan_algo.get_negative_prompt_embeds_cached(
                batch_size=eval_bs, device=accelerator.device
            )

        # Define wandb loggers outside
        logged_videos_wandb = None
        logged_prompts_wandb = None
        logged_rewards_wandb = None

        for step, batch in enumerate(
            tqdm(
                eval_loader,
                desc="Eval",
                disable=not accelerator.is_local_main_process,
            )
        ):
            if eval_num_batches is not None and step >= eval_num_batches:
                break
            batch = _to_device(batch, accelerator.device)
            batch = wan_algo.prepare_embeds(batch)

            # === NEW: convert List[Tensor] -> Tensor [B, text_len, C] ===
            if isinstance(batch.get("prompt_embeds", None), (list, tuple)):
                text_len = accelerator.unwrap_model(wan_algo.model).text_len  # WanModel里定义的固定长度
                batch["prompt_embeds"] = WanTextToVideo.pad_text_context_to_tensor(batch["prompt_embeds"], text_len)
            
            scheduler, timesteps = wan_algo.build_scheduler(is_training=False)
            eval_num_steps = cfg.sample.get("eval_num_steps", cfg.sample.num_steps)
            timesteps = timesteps[:eval_num_steps]

            videos, _, _, _ = trainer.rollout_batch(
                batch,
                scheduler,
                timesteps,
                neg_prompt_embeds=eval_neg_prompt_embeds,
                determistic=True,
            )

            rewards_dict, _ = reward_fn(
                videos, batch.get("prompts", None), [{"phase": "eval", "step": global_step, "epoch": epoch, "rank": accelerator.process_index} for _ in range(len(videos))]
            )
            for key, value in rewards_dict.items():
                value_tensor = torch.as_tensor(
                    value, device=accelerator.device, dtype=torch.float32
                )
                all_rewards[key].append(accelerator.gather(value_tensor).cpu().numpy())

            # === 1. Save ALL videos locally (All Ranks) ===
            # Structure: visualizations/eval/epoch_X_step_Y/rank_X_batch_Y_vid_Z.mp4
            vis_dir = os.path.join(self.output_dir, "visualizations", "eval", f"epoch_{epoch}_step_{global_step}")
            os.makedirs(vis_dir, exist_ok=True)
            
            videos_uint8 = self._videos_to_uint8(videos)
            for idx, v_frames in enumerate(videos_uint8):
                 fname = f"rank_{accelerator.process_index}_batch_{step}_vid_{idx}.mp4"
                 save_path = os.path.join(vis_dir, fname)
                 imageio.mimsave(save_path, v_frames, fps=eval_fps, codec="libx264", format="FFMPEG")

            # === 2. Cache first batch for WandB (Main Process Only) ===
            if logged_videos_wandb is None:
                logged_videos_wandb = videos.detach()
                logged_prompts_wandb = batch.get("prompts", None)
                logged_rewards_wandb = rewards_dict

        # === 3. Log to WandB (Main Process) ===
        if accelerator.is_main_process and logged_videos_wandb is not None:
             videos_uint8 = self._videos_to_uint8(logged_videos_wandb)
             num_samples = min(eval_num_videos, len(videos_uint8))
             
             with tempfile.TemporaryDirectory() as tmpdir:
                captions = []
                for idx in range(num_samples):
                    # Save temp for WandB upload
                    frames = [f for f in videos_uint8[idx]]
                    imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=eval_fps, codec="libx264", format="FFMPEG")
                    
                    prompt = (logged_prompts_wandb[idx] if isinstance(logged_prompts_wandb, (list, tuple)) else "")
                    reward_str = ""
                    if isinstance(logged_rewards_wandb, dict):
                        reward_str = " " + " | ".join(f"{k}: {float(logged_rewards_wandb[k][idx]):.2f}" for k in logged_rewards_wandb)
                    captions.append((prompt + reward_str).strip())
                    
                accelerator.log(
                    {
                        "eval/videos": [
                            wandb.Video(os.path.join(tmpdir, f"{idx}.mp4"), caption=captions[idx], format="mp4", fps=eval_fps)
                            for idx in range(num_samples)
                        ]
                    },
                    step=global_step,
                )

        if all_rewards:
            all_rewards = {k: np.concatenate(v) for k, v in all_rewards.items()}
            if accelerator.is_main_process:
                accelerator.log(
                    {f"eval/reward_{k}": float(np.mean(v)) for k, v in all_rewards.items()},
                    step=global_step,
                )

        wan_algo.model.train()

    def _save_checkpoint(
        self,
        accelerator: Accelerator,
        save_root: str,
        ema,
        global_step: int,
        epoch: int,
        wan_algo=None,
        reward_info=None,
    ):
        if accelerator.is_main_process:
            os.makedirs(save_root, exist_ok=True)
        accelerator.wait_for_everyone()
        
        # 1. Save Full Training State (For Resume)
        accelerator.save_state(save_root)
        
        # 2. Save EMA
        if accelerator.is_main_process and ema is not None:
            torch.save(
                {
                    "ema": ema.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                },
                os.path.join(save_root, "ema.pt"),
            )
            if reward_info is not None:
                torch.save(
                    reward_info,
                    os.path.join(save_root, "reward_info.pt"),
                )

        # 保存 reward_info（可读 JSON）
        if accelerator.is_main_process and reward_info is not None:
            reward_info = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in reward_info.items()
            }
            with open(os.path.join(save_root, "reward_info.json"), "w") as f:
                json.dump(reward_info, f, indent=2)

        # 3. Save Lightweight LoRA Adapter (For Inference)
        if wan_algo is not None:
            logger.info("Saving LoRA weights...")
            # Use accelerator to gather full state dict (handles FSDP gathering automatically)
            # NOTE: This loads full model to CPU RAM. For 14B model ~28GB RAM needed.
            full_state_dict = accelerator.get_state_dict(wan_algo.model)
            
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(wan_algo.model)
                
                # Check if it is a PeftModel or has save_pretrained
                save_func = getattr(unwrapped_model, "save_pretrained", None)
                if save_func is not None:
                    lora_dir = os.path.join(save_root, "lora_adapter")
                    # Pass the full state dict; PeftModel.save_pretrained will filter for lora keys internally
                    save_func(lora_dir, state_dict=full_state_dict)
                    logger.info(f"LoRA adapter saved to {lora_dir}")
                else:
                    logger.warning("Model is not a PeftModel, skipping separate LoRA save.")


    def training(self):        
        """
        Main RL Training Loop using Accelerate
        """
        cfg = self.cfg.training # Experiment config  "exp_flow_grpo.yaml" here
        algo_cfg = self.root_cfg.algorithm # Algo config  "wan_i2v.yaml" here"

        # Setup FSDP Plugin if requested
        fsdp_plugin = None
        if cfg.get("strategy") == "fsdp":
            # Default to bf16 mixed precision for FSDP if selected
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=ModuleWrapPolicy([WanAttentionBlock]),
                mixed_precision_policy=mp_policy,
                use_orig_params=True, # Critical for LoRA
            )

        accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=cfg.get("precision", "bf16"),
            # 对齐 flow_grpo：accumulation 需要乘 num_train_timesteps（否则更新频率会变）
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps * int(cfg.sample.num_steps * cfg.train.timestep_fraction),
            fsdp_plugin=fsdp_plugin,
        )

        if fsdp_plugin is not None:
            logger.info("Initialized FSDP Plugin with WanAttentionBlock wrap policy.")

        set_seed(cfg.seed, device_specific=True)

        if accelerator.is_main_process:
            accelerator.init_trackers("flow_grpo_rl", config=OmegaConf.to_container(cfg, resolve=True))
        
        # 1) build algo/model
        # Force training mode for WanAlgo (which defaults to inference if no Trainer attached)
        OmegaConf.set_struct(self.root_cfg.algorithm, False)
        self.root_cfg.algorithm.force_training = True
        OmegaConf.set_struct(self.root_cfg.algorithm, True)

        self._build_algo()
        
        # Set props on actual algo
        self.algo._device = accelerator.device
        self.algo._dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16
        
        # Wrap
        wan_algo = WanAlgoWrapper(self.algo)

        # Move components to device explicitely (Critical for VAE/CLIP/TextEncoder which are not passed to accelerator.prepare)
        logger.info(f"Moving VAE and Encoders to {accelerator.device}...")
        if hasattr(wan_algo, "vae"):
            wan_algo.vae.to(accelerator.device)
        if hasattr(wan_algo, "text_encoder") and wan_algo.text_encoder is not None:
            wan_algo.text_encoder.to(accelerator.device)
        if hasattr(wan_algo, "clip") and wan_algo.clip is not None:
            wan_algo.clip.to(accelerator.device)
        
        # Ensure VAE scale buffers are also on the right device/dtype
        if hasattr(wan_algo.algo, "vae_scale"):
            wan_algo.algo.vae_scale = [s.to(device=accelerator.device, dtype=wan_algo.algo._dtype) for s in wan_algo.algo.vae_scale]

        # freeze（对齐 flow_grpo）
        wan_algo.vae.requires_grad_(False)

        if getattr(wan_algo, "text_encoder", None) is not None and (not wan_algo.cfg.load_prompt_embed):
            wan_algo.text_encoder.requires_grad_(False)

        # 2) optimizer

        # Manual Optimizer Setup (aligned with train_wan2_1.py)
        transformer_trainable_parameters = [p for p in wan_algo.model.parameters() if p.requires_grad]
        
        use_8bit_adam = cfg.get("use_8bit_adam", False)
        optimizer_cls = torch.optim.AdamW
        
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                print("Please install bitsandbytes to use 8-bit Adam. Fallback to AdamW.")
        
        optimizer = optimizer_cls(
            transformer_trainable_parameters,
            lr=cfg.lr,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
            weight_decay=cfg.get("adam_weight_decay", 1e-4),
            eps=cfg.get("adam_epsilon", 1e-8),
        )

        # 3) dataset + sampler（对齐 flow_grpo：k = num_image_per_prompt）
        train_dataset = self._build_dataset("training")
        eval_dataset = self._build_dataset("validation")

        ##TODO: 这里需要和flow GRPO 的 sampler 对齐
        sampler = DistributedKRepeatSampler(
            train_dataset,
            batch_size=cfg.sample.sample_batch_size,              # per-gpu bs
            k=cfg.sample.num_image_per_prompt,                   # group size（全局）
            num_replicas=accelerator.num_processes,              # gpus number
            rank=accelerator.process_index,
            seed=cfg.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.data.num_workers,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
        )
        eval_batch_size = cfg.sample.get("eval_batch_size", None)
        if eval_batch_size is None:
            eval_batch_size = getattr(self.cfg, "validation", {}).get(
                "batch_size", cfg.sample.sample_batch_size
            )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=eval_dataset.collate_fn if hasattr(eval_dataset, "collate_fn") else None,
        )

        # 4) accelerate prepare
        # NOTE: We DO NOT prepare train_loader because it uses a custom infinite DistributedKRepeatSampler 
        # whose state (epoch) we manually control via the `sampler` variable variables. 
        # Accelerate wrapping can break this reference link or perform deep copies.
        # We manually handle device placement in the loop, so this is safe.
        wan_algo.model, optimizer, eval_loader = accelerator.prepare(
            wan_algo.model, optimizer, eval_loader
        )

        # ================= 🕵️‍♂️ 参数引用一致性检查 (CRITICAL CHECK) =================
        if accelerator.is_main_process:
            print("\n" + "="*50)
            print("[DEBUG] Checking Optimizer vs Model Parameter Linking...")
            
            # 1. 从当前用于训练的 FSDP 模型中，找一个 LoRA 参数
            # 注意：经过 prepare 后，wan_algo.model 已经是 FSDP 包裹的模型了
            model_param = None
            param_name = ""
            
            # 遍历 FSDP 模型参数
            for n, p in wan_algo.model.named_parameters():
                if "lora_B" in n and p.requires_grad:
                    model_param = p
                    param_name = n
                    break
            
            if model_param is None:
                print("❌ ERROR: Could not find any LoRA B parameter in wan_algo.model!")
            else:
                # 获取这个参数在内存中的真实 ID
                model_pid = id(model_param)
                print(f"  > Model Param: {param_name}")
                print(f"  > Memory ID (Model): {model_pid}")
                print(f"  > Requires Grad: {model_param.requires_grad}")

                # 2. 检查 Optimizer 到底在更新谁？
                # 收集 Optimizer 里所有参数组的参数 ID
                optimizer_pids = set()
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        optimizer_pids.add(id(p))
                
                # 3. 终极比对
                if model_pid in optimizer_pids:
                    print(f"  ✅ PASS: Optimizer IS tracking the correct parameter.")
                else:
                    print(f"  ❌ FAIL: Optimizer is tracking a DIFFERENT parameter object!")
                    print(f"     This explains why weights don't update.")
                    print(f"     Optimizer holds stale reference to pre-FSDP parameters.")
                    
                    # 尝试找一下旧参数的 ID (如果有办法访问原始模型的话)
                    # print(f"     Model Param ID not found in {len(optimizer_pids)} optimizer params.")

            print("="*50 + "\n")
        # =======================================================================

        # 5) reward_fn_：IDM Smoothness Reward
        # Import dynamically or use standard import if added at top
        from flow_grpo.idm_reward import idm_smoothness_score
        
        logger.info("Initializing IDM Smoothness Reward...")
        # Save plots to 'idm_plots' inside the experiment log directory
        save_dir = os.path.join(self.output_dir, "idm_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine checkpoint path (allow override from config, else default)
        checkpoint_path = cfg.get("reward_model_path", "/data/dex/vidar/vidar_ckpts/resnet_plus_robotwin/big_view.pt")
        
        reward_fn = idm_smoothness_score(accelerator.device, checkpoint_path=checkpoint_path, save_dir=save_dir)
 
        # 6) EMA（可选，对齐 flow_grpo）
        ema = None
        if cfg.train.ema and accelerator.is_main_process:
            trainable = list(filter(lambda p: p.requires_grad, wan_algo.model.parameters()))
            ema = EMAModuleWrapper(trainable, decay=cfg.train.ema_decay, update_step_interval=cfg.train.ema_interval, device=accelerator.device)

        trainer = FlowGRPOTrainer(wan_algo, reward_fn, cfg, accelerator)

        # neg embeds（对齐 flow_grpo：可以缓存一次；如果 no-cfg 就置 None）
        neg_prompt_embeds = None
        if cfg.train.cfg:
            neg_prompt_embeds = wan_algo.get_negative_prompt_embeds_cached(
                batch_size=cfg.sample.sample_batch_size, device=accelerator.device
            )

        # reward 异步 executor（对齐 flow_grpo）
        executor = futures.ThreadPoolExecutor(max_workers=cfg.reward.num_workers)

        global_step = 0
        train_iter = iter(train_loader)
        eval_every_n_epochs = cfg.get("eval_every_n_epochs", 1)
        eval_num_batches = cfg.get("eval_num_batches", 1)
        eval_num_videos = cfg.get("eval_num_videos", 4)
        eval_fps = cfg.get("eval_fps", 8)
        save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        save_root = os.path.join(str(self.output_dir), "checkpoints")
        
        # Track best reward for saving
        best_reward = -float("inf")

        for epoch in range(cfg.num_epochs):
            if eval_every_n_epochs and epoch % eval_every_n_epochs == 0 and epoch > 0:
                self._run_eval(
                    trainer,
                    wan_algo,
                    eval_loader,
                    reward_fn,
                    cfg,
                    accelerator,
                    global_step,
                    eval_num_batches=eval_num_batches,
                    eval_num_videos=eval_num_videos,
                    eval_fps=eval_fps,
                    epoch=epoch,
                )
            if save_every_n_epochs and epoch % save_every_n_epochs == 0 and epoch >= 10:
                ckpt_dir = os.path.join(save_root, f"epoch-{epoch}-step-{global_step}")
                self._save_checkpoint(
                    accelerator,
                    ckpt_dir,
                    ema,
                    global_step=global_step,
                    epoch=epoch,
                    wan_algo=wan_algo,
                )

            # Sampling（对齐 flow_grpo：每 epoch 采样 num_batches_per_epoch 次）
            samples = []
            for i in tqdm(range(cfg.sample.num_batches_per_epoch), disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}: sampling"):
                sampler.set_epoch(epoch * cfg.sample.num_batches_per_epoch + i)
                batch = next(train_iter)  # 无限 sampler
                batch = _to_device(batch, accelerator.device)

                with torch.no_grad():
                    batch = wan_algo.prepare_embeds(batch)  # 你的原逻辑：产出 prompt_embeds / clip_embeds / image_embeds / prompts 等
                    # 直接利用封装好的 Tokenizer 获取 prompt_ids
                        # === NEW: convert List[Tensor] -> Tensor [B, text_len, C] ===
                    if isinstance(batch.get("prompt_embeds", None), (list, tuple)):
                        text_len = accelerator.unwrap_model(wan_algo.model).text_len  # WanModel里定义的固定长度
                        batch["prompt_embeds"] = WanTextToVideo.pad_text_context_to_tensor(batch["prompt_embeds"], text_len)

                    if "prompt_ids" not in batch and "prompts" in batch:
                        # wan_algo.tokenizer 是 HuggingfaceTokenizer 的实例
                        # 调用它会自动触发 __call__ 方法，返回 input_ids
                        # 注意：它内部会自动处理 padding, truncation 和转 tensor
                        batch["prompt_ids"] = wan_algo.tokenizer(
                            batch["prompts"]
                        ).to(accelerator.device)
                # 这里是故意的，因为前两个epoch收集的group size会有bug,经过两个epoch后，group_size稳定成指定的
                if epoch < 2:
                    continue   
                # Build scheduler + timesteps
                scheduler, timesteps = wan_algo.build_scheduler(is_training=False)
                timesteps = timesteps[:cfg.sample.num_steps]  # 确保步数一致
                ###TODO: 之前推理 sample step 是40步，我们训练用多少步合适？

                with torch.no_grad():
                    videos, lat_traj, old_logps, kl = trainer.rollout_batch(
                        batch, scheduler, timesteps, neg_prompt_embeds=neg_prompt_embeds
                    )
                ### DEBUG: 看这里返回的video 是否正常
                print(f"DEBUG: videos shape: {videos.shape}, dtype: {videos.dtype}")
                
                # === Save Training Videos (All Ranks, All Samples) ===
                vis_train_dir = os.path.join(self.output_dir, "visualizations", "train", f"epoch_{epoch}_step_{global_step}")
                os.makedirs(vis_train_dir, exist_ok=True)
                
                # Convert
                vid_train_u8 = self._videos_to_uint8(videos)
                for v_idx, v_frames in enumerate(vid_train_u8):
                     fname = f"rank_{accelerator.process_index}_sample_{i}_vid_{v_idx}.mp4"
                     imageio.mimsave(os.path.join(vis_train_dir, fname), v_frames, fps=16, codec="libx264", format="FFMPEG")


                # Inject step info into metadata for better plotting filenames
                meta_list = batch.get("metadata", [])
                # Handle list of dicts or dict of lists? Usually collate_fn makes it list of dicts or dict of lists.
                # In train_wan2_1.py, collate returns dict of list if using standard collation for dicts?
                # Actually accelerated dataloader usually stacks tensors, but for dicts inside list?
                # Using batch.get("metadata", {}) implies it might be a dict.
                # Let's assume it's a list of dicts or we can construct a list of dicts to pass.
                # But reward_fn expects 'metadata' as 3rd arg.
                
                # To be safe, let's create a new metadata dict for reward_fn
                current_metadata = {"step": global_step, "epoch": epoch, "phase": "train", "rank": accelerator.process_index, "sample_idx": i}
                # Pass this as a single dict? Or list of dicts?
                # idm_reward __call__ receives it. 
                # Let's pass a list of dicts matching batch size if necessary, or just one dict if it broadcasts.
                # But flow_grpo logic might expect per-sample metadata.
                # Let's just update the list if it exists.
                
                # Construct metadata list for the batch
                batch_metadata = [current_metadata.copy() for _ in range(videos.shape[0])]
                
                # reward async（对齐 flow_grpo）
                fut = executor.submit(reward_fn, videos, batch.get("prompts", None), batch_metadata)
                time.sleep(0)

                # Store Future
                samples.append({
                    "prompt_embeds": batch["prompt_embeds"].detach().cpu(),   # [B, text_len, C] 放CPU省显存
                    "negative_prompt_embeds": (neg_prompt_embeds[:batch["prompt_embeds"].shape[0]] if neg_prompt_embeds is not None else None),
                    "clip_fea": batch.get("clip_embeds", None),
                    "y": batch.get("image_embeds", None),
                    "prompts": batch.get("prompts", []), 
                    "prompt_ids": batch.get("prompt_ids", None), # 必须要有 IDs 才能做完美 gather
                    "latents": lat_traj[:, :-1].detach().cpu(), # 移至 CPU 节省显存，训练前移回
                    "gt_latents": batch["video_lat"].detach().cpu(), # 为了给训练的时候覆盖第一帧latent
                    "next_latents": lat_traj[:, 1:].detach().cpu(),
                    "old_log_probs": old_logps.detach().cpu(),
                    "kl": kl.detach().cpu(),
                    "reward_fut": fut,
                })


            # --- STAT TRACKER WARMUP ---
            # 前两个 epoch 只收集数据更新 tracker，不训练，让 mean/std 稳定
            if epoch < 2 and cfg.per_prompt_stat_tracking:
                logger.info(f"Epoch {epoch} < 2: Collecting stats only...")
                # 即使是 warmup 也要把 reward 拿回来算一遍 tracker update
                # (逻辑省略，建议简单 continue，或者跑一遍 compute_advantages 但不执行 train_on_samples)
                # 为了简单起见，这里假设你需要完整跑一遍流程但不 backward，或者直接 continue
                # Flow-GRPO 源码是 continue，但前提是它在前 2 epoch 也会 gather reward。
                # 鉴于比较复杂，这里先保持 continue，但在 continue 前需要把 future 消费掉以免积压
                for s in samples:
                    s["reward_fut"].result()
                continue


            # --- GATHER & PROCESS PHASE ---
            all_train_samples = []
            
            for s in tqdm(samples, desc="Processing Rewards"):
                rewards_dict, _ = s["reward_fut"].result()
                
                # s['kl'] is on CPU, move to GPU for calc
                kl_gpu = s["kl"].to(accelerator.device)
                
                # Reward Calculation: Avg - KL Penalty
                if "avg" in rewards_dict:
                    r_raw = rewards_dict["avg"]
                else:
                    # Fallback: sum all numerical list values (assuming single or multi-objective without 'avg')
                    # This handles cases like video_ocr_score returning {'video_ocr': scores}
                    keys = [k for k in rewards_dict.keys() if isinstance(rewards_dict[k], (list, np.ndarray))]
                    if not keys:
                         raise ValueError(f"No valid reward scores found in keys: {rewards_dict.keys()}")
                    r_raw = np.array(rewards_dict[keys[0]])
                    for k in keys[1:]:
                        r_raw += np.array(rewards_dict[k])
                
                r_tensor = torch.as_tensor(r_raw, device=accelerator.device, dtype=torch.float32)
                
                # KL usually summed/meaned over timesteps for penalty
                kl_penalty = s["kl"].float().to(accelerator.device).mean(dim=1) # [B]
                r_final = r_tensor - cfg.sample.kl_reward * kl_penalty
                
                s["rewards"] = r_final.unsqueeze(-1) # [B, 1]
                del s["reward_fut"]
                
                # Move everything back to GPU for training preparation or keep on CPU for collation
                # 建议先 keep on CPU，gather 时再动
                all_train_samples.append(s)

            # Collate all samples from this rank
            # 我们把所有 batch 拼起来形成一个大 Batch
            cat = {}
            keys = ["prompt_embeds", "negative_prompt_embeds", "clip_fea", "y",
                    "latents", "next_latents", "old_log_probs", "kl", "rewards" ,"gt_latents"]
            for k in keys:
                # 注意处理 None
                if all_train_samples[0].get(k) is not None:
                     # 此时 tensor 可能在 CPU 或 GPU，建议统一转到 GPU
                    tensors = [s[k].to(accelerator.device) for s in all_train_samples]
                    cat[k] = torch.cat(tensors, dim=0)
                else:
                    cat[k] = None

            # --- GLOBAL GATHER (CRITICAL FOR GRPO) ---
            # 1. Gather Rewards
            local_rewards = cat["rewards"] # [Local_B, 1]
            gathered_rewards = accelerator.gather(local_rewards).detach().cpu().numpy() # [Global_B, 1]
            
            # Compute Mean Reward on ALL processes (deterministically)
            # Since gather returns the same tensor on all ranks, this should be consistent.
            current_mean_reward = gathered_rewards.mean()

            if accelerator.is_main_process:
                # logger.info(f"Step {global_step}: Mean Reward = {mean_reward:.4f}")
                accelerator.log({"train/mean_reward": current_mean_reward}, step=global_step)
            
            # Save Best Checkpoint Logic
            if current_mean_reward > best_reward:
                old_best = best_reward
                best_reward = current_mean_reward
                if accelerator.is_main_process:
                    logger.info(f"🌟 New Best Reward: {best_reward:.4f} (Was {old_best:.4f}). Saving best checkpoint...")
                
                best_save_dir = os.path.join(save_root, "best_reward")
                self._save_checkpoint(
                    accelerator,
                    best_save_dir,
                    ema,
                    global_step=global_step,
                    epoch=epoch,
                    wan_algo=wan_algo,
                    reward_info={"mean_reward": best_reward, "epoch": epoch, "step": global_step},
                )
            
            # 2. Gather Prompts (Need IDs for accuracy)
            # 假设 batch 里有 "prompt_ids" (Tensor [B, L])
            # 如果没有 prompt_ids，只能 gather objects (strings)，这非常慢且容易 OOM
            prompts_local_ids = None
            if all_train_samples[0].get("prompt_ids") is not None:
                ids_list = [s["prompt_ids"].to(accelerator.device) for s in all_train_samples]
                prompts_local_ids = torch.cat(ids_list, dim=0)
                gathered_ids = accelerator.gather(prompts_local_ids).cpu().numpy()
                # Decode IDs back to strings (Optional, if tracker accepts IDs)
                # 这里假设 tracker 需要 list of strings
                prompts_gathered = wan_algo.tokenizer.tokenizer.batch_decode(gathered_ids, skip_special_tokens=True)
            else:
                # Fallback: 只用本地 prompt (会导致各卡 normalization 不一致，性能下降)
                logger.warning("No prompt_ids found! Using local prompts only for Advantage Normalization.")
                prompts_local_list = sum([s["prompts"] for s in all_train_samples], [])
                prompts_gathered = prompts_local_list # Error: logic mismatch if using gather rewards


            # --- COMPUTE ADVANTAGES ---
            if cfg.per_prompt_stat_tracking and prompts_gathered is not None:
                advantages_np, group_size, _ = trainer.compute_advantages(prompts_gathered, gathered_rewards)
            else:
                # Fallback global norm
                advantages_np, _, _ = trainer.compute_advantages(None, gathered_rewards)

            # --- UN-GATHER ---
            # 取回属于当前 rank 的那部分 advantage
            adv_tensor = torch.as_tensor(advantages_np, device=accelerator.device, dtype=torch.float32) # [Global_B, 1]
            
            # Reshape to [Num_Processes, Local_B, 1] -> [Local_B, 1]
            # 注意：accelerator.gather 会自动 padding 数据以对其，需要小心截断
            # 简单做法：直接根据 rank 切片 (假设数据量均匀)
            local_batch_size = cat["rewards"].shape[0]
            # 计算 offset (不完美，建议用 accelerator.gather 的 padding 逻辑处理，或假设整除)
            # 这里假设 total_samples % num_replicas == 0
            start_idx = accelerator.process_index * local_batch_size
            end_idx = start_idx + local_batch_size
            local_adv = adv_tensor[start_idx:end_idx]
            
            # Broadcast Advantage to [B, S]
            S = cat["old_log_probs"].shape[1]
            local_adv = local_adv.expand(-1, S) # [B, S]

            # --- PREPARE FOR TRAIN ---
            train_payload = {
                "prompt_embeds": cat["prompt_embeds"],
                "negative_prompt_embeds": cat["negative_prompt_embeds"],
                "clip_fea": cat["clip_fea"],
                "y": cat["y"],
                "latents": cat["latents"],
                "next_latents": cat["next_latents"],
                "old_log_probs": cat["old_log_probs"],
                "advantages": local_adv,
                "gt_latents": cat["gt_latents"],
            }
            
            # --- INNER EPOCH TRAINING ---
            for inner_epoch in range(cfg.train.num_inner_epochs):
                 # Re-build scheduler for training (noise scheduling)
                scheduler, timesteps = wan_algo.build_scheduler(is_training=False)
                # Move scheduler timesteps to device to avoid runtime error in index_for_timestep
                scheduler.timesteps = scheduler.timesteps.to(accelerator.device)
                timesteps = scheduler.timesteps[:cfg.sample.num_steps]
                
                info = trainer.train_on_samples(
                    train_payload, scheduler, timesteps, optimizer, global_step=global_step, ema=ema
                )
                
                if accelerator.is_main_process and info:
                    accelerator.log({f"train/{k}": v.item() for k, v in info.items()}, step=global_step)
            
                global_step += 1




from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import sde_step_with_logprob

class WanAlgoWrapper:
    """
    Wrapper for WanAlgo that adds Flow-GRPO specific methods (sampling with logprob, etc.)
    Delegates everything else to the underlying algo.
    """
    def __init__(self, algo):
        self.algo = algo

    def __getattr__(self, name):
        return getattr(self.algo, name)

    def format_timestep(self, t, B, device):
        # Ensure t is a tensor of shape [B]
        if torch.is_tensor(t):
            t = t.to(device)
            if t.ndim == 0:
                t = t.expand(B)
            elif t.ndim == 1 and len(t) == 1:
                t = t.expand(B)
            return t.long()
        else:
            return torch.full((B,), t, device=device, dtype=torch.long)

    # def prepare_initial_latents(self, batch_size, device, dtype, batch):
    #     # T2V/I2V logic: Start from Gaussian noise

    #     video_pred_lat = torch.randn(
    #         batch_size, 
    #         self.algo.lat_c, 
    #         self.algo.lat_t, 
    #         self.algo.lat_h, 
    #         self.algo.lat_w, 
    #         device=device, 
    #         dtype=dtype
    #     )
    #     hist_len = 1
    #     if self.diffusion_forcing.enabled:
    #             video_pred_lat[:, :, :hist_len] = video_lat[:, :, :hist_len]
    #             t_expanded = torch.full((batch_size, self.lat_t), t, device=self.device)
    #             t_expanded[:, :hist_len] = self.inference_timesteps[-1]
    #     else:
    #             t_expanded = torch.full((batch_size,), t, device=self.device)

    def construct_diffusion_forcing_timesteps(self, batch_size, t_curr, device, clean_t, hist_len=1):
        """
        统一构建时间向量/矩阵，处理 Diffusion Forcing 逻辑。
        
        Args:
            batch_size: 当前批次大小
            t_curr: 当前生成步的时间 (High Noise)
            device: 设备
            clean_t: 纯净/历史帧的时间 (No Noise, 通常是 timesteps[-1])
            hist_len: 历史帧长度 (默认 1)
        
        Returns:
            t_vec: [B] (Normal) 或 [B, F] (Diffusion Forcing)
        """
        # 访问底层的 diffusion_forcing 配置
        if self.algo.diffusion_forcing.enabled:
            # 获取 latent 时间维度长度 (F)
            # 假设 WanModel 的 latent_t 属性存在
            num_frames = self.algo.lat_t 
            
            # 1. 初始化 [B, F] 矩阵，全填 t_curr
            t_vec = torch.full(
                (batch_size, num_frames), 
                t_curr, 
                device=device, 
            )
            
            # 2. 将历史帧强制设为 clean_t
            t_vec[:, :hist_len] = clean_t
            
            return t_vec
        else:
            # 正常模式：使用 format_timestep 广播标量
            return self.format_timestep(t_curr, batch_size, device)

    def _predict_noise(self, latents, t_vec, prompt_embeds, negative_prompt_embeds, do_cfg, guidance_scale, clip_fea=None, y=None):
        def run_forward(lat, t, context):
            # Matches WanModel forward signature: x, t, context, seq_len, clip_fea, y
            return self.model(
                lat,
                t=t,
                context=context,
                seq_len=self.algo.max_tokens,
                clip_fea=clip_fea,
                y=y
            )

        if do_cfg:
            #TODO: cfg should align with wan algo cfg logic in wan_t2v.py
            noise_uncond = run_forward(latents, t_vec, negative_prompt_embeds)
            noise_text = run_forward(latents, t_vec, prompt_embeds)
            return noise_uncond + guidance_scale * (noise_text - noise_uncond)
        else:
            return run_forward(latents, t_vec, prompt_embeds)

    @torch.no_grad()
    def sample_step_with_logprob(self, scheduler, latents, t_vec, prompt_embeds, negative_prompt_embeds,
                                 guidance_scale, do_cfg, clip_fea=None, y=None, determistic=False, debug_label=None):
        """
        Performs one sampling step and returns log_prob of the taken action (noise).
        """
        noise = self._predict_noise(
            latents, t_vec, prompt_embeds, negative_prompt_embeds, 
            do_cfg, guidance_scale, clip_fea, y
        )
        
        # SDE Step (Generation Mode: prev_sample=None)
        t_scheduler = t_vec
        if t_scheduler.ndim > 1:
            t_scheduler = t_scheduler[:, -1]

        if debug_label:
             print(f"\n[{debug_label}] Inside sample_step_with_logprob:")
             print(f"  > latents hash: {latents.float().sum().item():.4f}")
             print(f"  > noise hash:   {noise.float().sum().item():.4f}")
             print(f"  > t_scheduler:  {t_scheduler[0] if t_scheduler.numel()>0 else '?'}")
             print(f"  > determistic:  {determistic}")

        prev, logp, prev_mean, std, dt = sde_step_with_logprob(
            scheduler,
            noise.float(),
            t_scheduler,
            latents.float(),
            prev_sample=None,
            determistic=determistic,
            return_dt_and_std_dev_t=True
        )

        if debug_label:
             print(f"  > logp value:   {logp[0].item():.6f}")
             print(f"  > prev_sample:  {prev.float().sum().item():.4f}")

        return prev, logp, prev_mean, std, dt

    def logprob_of_transition(self, scheduler, latents, next_latents, t_vec, prompt_embeds, negative_prompt_embeds,
                              guidance_scale, do_cfg, clip_fea=None, y=None, debug_label=None):
        """
        Calculates log_prob of a transition (latents -> next_latents) given the policy.
        """
        noise = self._predict_noise(
            latents, t_vec, prompt_embeds, negative_prompt_embeds, 
            do_cfg, guidance_scale, clip_fea, y
        )
        
        # SDE Step (Training Mode: prev_sample=next_latents)
        t_scheduler = t_vec
        if t_scheduler.ndim > 1:
            t_scheduler = t_scheduler[:, -1]

        if debug_label:
             print(f"\n[{debug_label}] Inside logprob_of_transition:")
             print(f"  > latents hash: {latents.float().sum().item():.4f}")
             print(f"  > next_latents: {next_latents.float().sum().item():.4f}")
             print(f"  > noise hash:   {noise.float().sum().item():.4f}")
             print(f"  > t_scheduler:  {t_scheduler[0] if t_scheduler.numel()>0 else '?'}")

        prev, logp, prev_mean, std, dt = sde_step_with_logprob(
            scheduler,
            noise.float(),
            t_scheduler,
            latents.float(),
            prev_sample=next_latents.float(),
            return_dt_and_std_dev_t=True
        )

        if debug_label:
             print(f"  > logp value:   {logp[0].item():.6f}")

        return prev, logp, prev_mean, std, dt
