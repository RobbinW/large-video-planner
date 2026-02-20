import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# from algorithms.cogvideo import CogVideoXImageToVideo, CogVideoXVAE
from algorithms.wan import WanImageToVideo, WanTextToVideo
from datasets.dummy import DummyVideoDataset
from datasets.openx_base import OpenXVideoDataset
from datasets.droid import DroidVideoDataset
from datasets.something_something import SomethingSomethingDataset
from datasets.epic_kitchen import EpicKitchenDataset
from datasets.pandas import PandasVideoDataset
from datasets.ego4d import Ego4DVideoDataset
from datasets.agibot_world import AgibotWorldDataset
from datasets.mixture import MixtureDataset
from datasets.robotwin import RobotwinDataset
from datasets.real_piper import RealPiperDataset
from datasets.video_base import SingleFrameVideoDataset
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        # cogvideox_i2v=CogVideoXImageToVideo,
        # cogvideox_vae=CogVideoXVAE,
        wan_i2v=WanImageToVideo,
        wan_t2v=WanTextToVideo,
        wan_toy=WanImageToVideo,
    )

    compatible_datasets = dict(
        mixture=MixtureDataset,
        mixture_robot=MixtureDataset,
        dummy=DummyVideoDataset,
        something_something=SomethingSomethingDataset,
        epic_kitchen=EpicKitchenDataset,
        pandas=PandasVideoDataset,
        ego4d=Ego4DVideoDataset,
        bridge=OpenXVideoDataset,
        droid=DroidVideoDataset,
        robotwin=RobotwinDataset,
        real_piper=RealPiperDataset,
        agibot_world=AgibotWorldDataset,
        language_table=OpenXVideoDataset,
        ours_test=SingleFrameVideoDataset,
        # austin_buds=OpenXVideoDataset,
        # austin_sailor=OpenXVideoDataset,
        # austin_sirius=OpenXVideoDataset,
        # bc_z=OpenXVideoDataset,
        # berkeley_autolab=OpenXVideoDataset,
        # berkeley_cable=OpenXVideoDataset,
        # berkeley_fanuc=OpenXVideoDataset,
        # cmu_stretch=OpenXVideoDataset,
        # dlr_edan=OpenXVideoDataset,
        # dobbe=OpenXVideoDataset,
        # fmb=OpenXVideoDataset,
        # fractal=OpenXVideoDataset,
        # iamlab_cmu=OpenXVideoDataset,
        # jaco_play=OpenXVideoDataset,
        # nyu_franka=OpenXVideoDataset,
        # roboturk=OpenXVideoDataset,
        # stanford_hydra=OpenXVideoDataset,
        # taco_play=OpenXVideoDataset,
        # toto=OpenXVideoDataset,
        # ucsd_kitchen=OpenXVideoDataset,
        # utaustin_mutex=OpenXVideoDataset,
        # viola=OpenXVideoDataset,
    )

    def _build_strategy(self):
        from lightning.pytorch.strategies.fsdp import FSDPStrategy

        if self.cfg.strategy == "ddp":
            return super()._build_strategy()
        elif self.cfg.strategy == "fsdp":
            # Fix: Only use device mesh for multi-node setups where HSDP is intended.
            # For single node (or when calculation doesn't match world size), use None.
            if self.cfg.num_nodes > 1:
                # Assuming 8 GPUs per node for calculation consistency with your hardware
                # Adjust logic if nodes have different GPU counts
                if self.cfg.num_nodes >= 8:
                    device_mesh = (self.cfg.num_nodes // 8, 32)
                else: 
                     # Original logic was (1, num_nodes * 4), presumably for 4-GPU nodes?
                     # Let's set it to None to be safe unless we are strictly doing HSDP
                     device_mesh = None
            else:
                device_mesh = None
                
            return FSDPStrategy(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                cpu_offload=True, # Enable CPU Offload to save GPU memory for VAE activations
                auto_wrap_policy=ModuleWrapPolicy(self.algo.classes_to_shard()),
                sharding_strategy="FULL_SHARD",
                # sharding_strategy="HYBRID_SHARD",
                device_mesh=device_mesh,
            )

        else:
            return self.cfg.strategy

    def download_dataset(self):
        dataset = self._build_dataset("training")
