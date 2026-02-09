import sys
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math

# Add path to IDM codebase
IDM_ROOT = "/data/dex/vidar/vidar_old/Vidar/vidar"
if IDM_ROOT not in sys.path:
    sys.path.append(IDM_ROOT)

try:
    from idm.idm import IDM
except ImportError:
    print(f"[Warning] Could not import IDM from {IDM_ROOT}. Please check sys.path or environment.")

def plot_actions(gt_actions, pred_actions, save_dir, sample_idx):
    """
    gt_actions: [T, 14] numpy array (can be all zeros if no GT)
    pred_actions: [T, 14] numpy array
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_steps = gt_actions.shape[0]
    num_dims = gt_actions.shape[1]

    # Calculate grid layout
    cols = 4
    rows = math.ceil(num_dims / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    time_steps = np.arange(num_steps)

    for i in range(num_dims):
        ax = axes[i]
        
        # Plot GT (only if not all zeros, or we can choose to skip it)
        if not np.all(gt_actions[:, i] == 0):
            ax.plot(time_steps, gt_actions[:, i], color='blue', label='Ground Truth', linewidth=1)
        
        # Plot Pred
        if i < pred_actions.shape[1]:
            ax.plot(time_steps, pred_actions[:, i], color='red', label='Predicted', linestyle='--', linewidth=1)
            
        ax.set_title(f'Dimension {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # Logic for y-axis limits
        vals = []
        vals.append(gt_actions[:, i])
        if i < pred_actions.shape[1]:
            vals.append(pred_actions[:, i])
        vals = np.concatenate(vals)

        # Standard range check
        if np.all(vals >= -1.2) and np.all(vals <= 1.2):
            ax.set_ylim(-1.2, 1.2)

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    save_name = f"sample_{sample_idx}_comparison.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    # print(f"Plot saved successfully to: {save_path}")

class IDMSmoothnessReward:
    def __init__(self, checkpoint_path, device, model_name="resnet_plus", num_frames=3, save_dir=None):
        self.device = device
        self.num_frames = num_frames
        self.save_dir = save_dir
        
        print(f"[IDM Reward] Initializing IDM model from {checkpoint_path} with num_frames={num_frames}...")
        try:
            # IDM model expects num_frames to define channel depth
            self.net = IDM(model_name=model_name, output_dim=14, num_frames=num_frames)
            
            # Load Checkpoint
            loaded_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if "model_state_dict" in loaded_dict:
                state_dict = loaded_dict["model_state_dict"]
            else:
                state_dict = loaded_dict
            
            # Load with strict=False
            self.net.load_state_dict(state_dict, strict=False)
            self.net.to(device)
            self.net.eval()
            print("[IDM Reward] Model loaded successfully.")
        except Exception as e:
            print(f"[IDM Reward] Error loading IDM model: {e}")


        # Preprocessing: Resize -> Normalize (ImageNet stats)
        # eval_idm.py uses DinoPreprocessor which defaults to 512x512
        self.resize = transforms.Resize((512, 512))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess_video(self, video_tensor):
        """
        Args:
            video_tensor: [T, C, H, W] tensor in [0, 1] range
        Returns:
            processed: [T, C, 512, 512] normalized tensor
        """
        video_tensor = self.resize(video_tensor)
        processed = self.normalize(video_tensor)
        return processed

    def predict_actions(self, video_tensor):
        """
        Infer actions from video tensor using IDM.
        Args:
            video_tensor: [T, C, H, W] in [0, 1]
        Returns:
            actions: [T, Action_Dim]
        """
        if self.net is None:
            return torch.zeros((video_tensor.shape[0], 14), device=self.device)

        processed_frames = self.preprocess_video(video_tensor) # [T, 3, 512, 512]
        T = processed_frames.shape[0]
        actions = []

        with torch.no_grad():
            for i in range(T):
                # Windowing logic [t-1, t, t+1]
                indices = []
                if self.num_frames == 1:
                    indices = [i]
                elif self.num_frames == 2:
                    indices = [i-1, i]
                elif self.num_frames == 3:
                    indices = [i-1, i, i+1]
                else:
                    indices = [i] * self.num_frames
                
                indices = [max(0, min(idx, T-1)) for idx in indices]
                
                stack = [processed_frames[idx] for idx in indices]
                img_input = torch.stack(stack).to(self.device).unsqueeze(0) # [1, num_frames, 3, 512, 512]
                
                output = self.net(img_input, None) # [1, 14]
                actions.append(output)

        return torch.cat(actions, dim=0)

    def compute_smoothness(self, actions):
        """
        Smoothness reward for 14D actions.

        actions: [T, 14]
        - joint dims are radians
        - dim 6 and 13 are grippers in [0,1]
        - fps = 16Hz => dt = 1/16

        Returns:
        score (float): higher is smoother, in [0, 10] approximately.
        """
        if actions.shape[0] < 4:
            return 0.0

        dt = 1.0 / 16.0
        T, D = actions.shape
        device, dtype = actions.device, actions.dtype

        # ---- dims ----
        grip_dims = [6, 13]

        # ---- per-dim speed limits (rad/s) ----
        # PiPER joint max speed (deg/s): J1 180, J2 195, J3 180, J4-6 225
        # -> rad/s: 3.1416, 3.4034, 3.1416, 3.9270, 3.9270, 3.9270
        v_arm = torch.tensor([3.1416, 3.4034, 3.1416, 3.9270, 3.9270, 3.9270],
                            device=device, dtype=dtype)
        safety = 0.85
        v_arm = v_arm * safety

        v_max = torch.zeros(D, device=device, dtype=dtype)
        v_max[0:6] = v_arm
        v_max[7:13] = v_arm
        v_max[6] = 2.0   # gripper units/s
        v_max[13] = 2.0

        # ---- per-dim accel limits (rad/s^2) ----
        a_max = torch.full((D,), 5.0, device=device, dtype=dtype)
        a_max[6] = 10.0  # gripper units/s^2
        a_max[13] = 10.0

        # ---- finite differences ----
        v = (actions[1:] - actions[:-1]) / dt      # [T-1, D]
        a = (v[1:] - v[:-1]) / dt                  # [T-2, D]
        j = (a[1:] - a[:-1]) / dt                  # [T-3, D]

        # ---- huber ----
        def huber(x, delta):
            ax = x.abs()
            return torch.where(ax <= delta, 0.5 * (x ** 2), delta * (ax - 0.5 * delta))

        # ---- dim weights ----
        w = torch.ones(D, device=device, dtype=dtype)
        w[6] = 0.2
        w[13] = 0.2
        w = w / (w.mean() + 1e-8)

        # ---- huber thresholds ----
        delta_a = a_max * 0.5                 # [D]
        delta_j = (a_max / dt) * 0.5          # [D]  (since jerk ~ Δa/dt)

        # ---- energy penalties ----
        # broadcast: a/j shape [..., D], delta_*/w shape [D]
        acc_pen  = (huber(a, delta_a) * w).mean()
        jerk_pen = (huber(j, delta_j) * w).mean()

        # ---- soft-limit penalties ----
        v_violate = torch.relu(v.abs() - v_max)
        a_violate = torch.relu(a.abs() - a_max)
        vlim_pen = ((v_violate ** 2) * w).mean()
        alim_pen = ((a_violate ** 2) * w).mean()

        # ---- total penalty ----
        penalty_raw = (
            0.5 * jerk_pen +
            0.25 * acc_pen +
            2.0 * vlim_pen +
            1.0 * alim_pen
        )

        # ---- debug prints (opt-in) ----
        # Set env: DEBUG_SMOOTHNESS=1 to print
        # with torch.no_grad():
        #     # basic stats to see scale
        #     v_max_abs = v.abs().max().item()
        #     a_max_abs = a.abs().max().item()
        #     j_max_abs = j.abs().max().item()

        #     v_mean = v.abs().mean().item()
        #     a_mean = a.abs().mean().item()
        #     j_mean = j.abs().mean().item()

        #     print(
        #         "[DEBUG] Smoothness components:\n"
        #         f"  penalty_raw = {penalty_raw.item():.4f}\n"
        #         f"    jerk_pen  = {jerk_pen.item():.4f}\n"
        #         f"    acc_pen   = {acc_pen.item():.4f}\n"
        #         f"    vlim_pen  = {vlim_pen.item():.4f}\n"
        #         f"    alim_pen  = {alim_pen.item():.4f}\n"
        #         f"  stats:\n"
        #         f"    max|v|={v_max_abs:.4f}, mean|v|={v_mean:.4f}\n"
        #         f"    max|a|={a_max_abs:.4f}, mean|a|={a_mean:.4f}\n"
        #         f"    max|j|={j_max_abs:.4f}, mean|j|={j_mean:.4f}\n"
        #     )

        # reference scale: roughly "good smoothness" level
        P0 = 2000.0      # try 2000 ~ 5000

        gamma = 0.5      # 0.5 is very safe for GRPO
        MaxReward = 10.0

        score = MaxReward * torch.pow(1.0 + penalty_raw / P0, -gamma)

        return score.item()


    def __call__(self, images, prompts, metadata):
        """
        Main entry point.
        """
        rewards = []
        
        # 1. Denormalize videos from [-1, 1] (Diffusion) to [0, 1] (IDM)
        images = (images.float() + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        bs, c, t, h, w = images.shape
        
        # Only plot the first video in the batch to avoid IO bottleneck
        should_plot = (self.save_dir is not None)
        
        for i in range(bs):
            video = images[i].permute(1, 0, 2, 3) # [T, C, H, W]
            
            actions = self.predict_actions(video)
            score = self.compute_smoothness(actions)
            rewards.append(score)
            
            if should_plot and i == 0:
                # Use metadata for unique naming if possible
                # E.g. prompt_id or global step
                # Fallback to random or just overwrite
                
                # Check if we have prompt text to sanitize for filename?
                # Or just use a hash/index
                prompt_snippet = prompts[i][:30].replace(" ", "_") if i < len(prompts) else "unknown"
                
                # Check metadata for step/phase info
                phase = "unknown"
                step_info = ""
                rank_str = ""
                
                if metadata and isinstance(metadata, list) and len(metadata) > i:
                    meta_item = metadata[i]
                    if isinstance(meta_item, dict):
                        phase = meta_item.get("phase", "unknown")
                        step = meta_item.get("step", "")
                        epoch = meta_item.get("epoch", "")
                        rank = meta_item.get("rank", None)
                        sample_idx = meta_item.get("sample_idx", None)

                        if step != "":
                            step_info = f"_step{step}"
                        if epoch != "":
                            step_info += f"_epoch{epoch}"
                        if sample_idx is not None:
                            step_info += f"_s{sample_idx}"
                        if rank is not None:
                            rank_str = f"_rank{rank}"

                # Append random int to avoid overwrite within same batch/step if multiple distinct prompts
                if rank_str:
                     # Deterministic naming with rank and sample_idx
                     sample_name = f"{phase}_video_{prompt_snippet}{step_info}{rank_str}"
                else:
                     # Fallback to random if rank/sample info missing
                     sample_name = f"{phase}_video_{prompt_snippet}{step_info}_{np.random.randint(0, 1000)}"
                
                gt = np.zeros_like(actions.cpu().numpy())
                plot_actions(gt, actions.cpu().numpy(), self.save_dir, sample_name)
            
        return {"idm_smoothness": np.array(rewards)}, metadata

def idm_smoothness_score(device, checkpoint_path="/data/dex/DanceGRPO/my_checkpoints/idm_best.pth", num_frames=3, save_dir=None):
    """
    Factory function for the reward.
    """
    scorer = IDMSmoothnessReward(checkpoint_path, device, num_frames=num_frames, save_dir=save_dir)
    
    def _fn(images, prompts, metadata):
        return scorer(images, prompts, metadata)
        
    return _fn
