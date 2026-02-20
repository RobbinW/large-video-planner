import random
import os
from pathlib import Path
import torch
import pandas as pd
import wandb
import time
from tqdm import trange
from torch.utils.data import IterableDataset
from datasets.dummy import DummyVideoDataset
from datasets.openx_base import OpenXVideoDataset
from datasets.droid import DroidVideoDataset
from datasets.something_something import SomethingSomethingDataset
from datasets.epic_kitchen import EpicKitchenDataset
from datasets.pandas import PandasVideoDataset
from datasets.ego4d import Ego4DVideoDataset
from datasets.mixture import MixtureDataset
from datasets.agibot_world import AgibotWorldDataset
from datasets.robotwin import RobotwinDataset
from datasets.real_piper import RealPiperDataset
from .exp_base import BaseExperiment


class ProcessDataExperiment(BaseExperiment):
    """
    An experiment class for you to easily process an existing
    dataset into another, by creating a new csv metadata file and new files.

    e.g. The `cache_prompt_embed` method illustrates caching the prompt embeddings and
    adding a field `prompt_embed_path` to a copy ofthe metadata csv.

    e.g. The `visualize_dataset` method illustrates visualizing a sample of videos from the dataset with their captions.

    Add your processing methods here, and follow README.md to run.
    """

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

    def _build_dataset(
        self, disable_filtering: bool = True, split: str = "all"
    ) -> torch.utils.data.Dataset:
        if disable_filtering:
            self.root_cfg.dataset.filtering.disable = True
        return self.compatible_datasets[self.root_cfg.dataset._name](
            self.root_cfg.dataset, split=split
        )

    def _get_save_dir(self, dataset: torch.utils.data.Dataset):
        save_dir = self.cfg.new_data_root
        if self.cfg.new_data_root is None:
            save_dir = self.output_dir / dataset.data_root.name
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def benchmark_dataloader(self):
        """Benchmark the speed of the dataloader."""
        cfg = self.cfg.benchmark_dataloader
        dataset = self._build_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )
        for i in trange(1000000):
            time.sleep(0.001)

    def visualize_dataset(self):
        """Visualize a sample of videos from the dataset with their captions.

        This method:
        1. Creates a dataloader for the dataset
        2. Logs the videos and their captions to wandb

        Sample command:
        python main.py +name=process_data experiment=process_data dataset=video_openx experiment.tasks=[visualize_dataset]
        """

        cfg = self.cfg.visualize_dataset
        dataset = self._build_dataset(
            disable_filtering=cfg.disable_filtering, split="training"
        )
        shuffle = not isinstance(dataset, IterableDataset)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=shuffle
        )

        log_dict = {}
        self._build_logger()

        samples_seen = 0
        for batch in dataloader:
            if samples_seen >= cfg.n_samples:
                break

            for i in range(len(batch["videos"])):
                if samples_seen >= cfg.n_samples:
                    break

                prompts = None
                if "prompts" in batch:
                    prompts = batch["prompts"][i]

                if cfg.use_processed:
                    video = batch["videos"][i]  # [T, C, H, W]
                    # Convert from [-1, 1] to [0, 255] and correct format for wandb
                    video = ((video + 1) / 2 * 255).clamp(0, 255)
                    video = video.to(torch.uint8).numpy()  # [T, H, W, C]
                    log_dict[f"sample_{samples_seen}"] = wandb.Video(
                        video, caption=prompts, fps=16
                    )
                else:
                    # Log raw video file
                    video_path = str(dataset.data_root / batch["video_path"][i])
                    log_dict[f"sample_{samples_seen}"] = wandb.Video(
                        video_path, caption=prompts, fps=16
                    )

                samples_seen += 1
                if samples_seen % 8 == 0:
                    wandb.log(log_dict)
                    log_dict = {}

        # Log any remaining samples
        if log_dict:
            wandb.log(log_dict)

    def cache_prompt_embed(self):
        """Cache prompt embeddings for all captions in the dataset.

        This method:
        1. Takes captions from the dataset metadata
        2. Generates T5 embeddings for each caption using CogVideo's T5 encoder
        3. Saves embeddings as .pt files alongside the videos
        4. Creates a new metadata CSV with an added 'prompt_embed_path' column

        Sample commands:
        # Cache embeddings for OpenX dataset:
        python main.py +name=process_data experiment=process_data dataset=video_openx experiment.tasks=[cache_prompt_embed]

        # Specify custom output directory:
        python main.py +name=process_data experiment=process_data dataset=video_openx experiment.tasks=[cache_prompt_embed] experiment.new_data_root=data/processed

        # Adjust batch size:
        python main.py +name=process_data experiment=process_data dataset=video_openx experiment.tasks=[cache_prompt_embed] experiment.cache_prompt_embed.batch_size=64
        """
        cfg = self.cfg.cache_prompt_embed
        batch_size = cfg.batch_size

        if self.cfg.num_nodes != 1:
            raise ValueError("This script only supports 1 node. ")

        if "wan" in self.root_cfg.algorithm._name:
            from algorithms.wan.modules.t5 import umt5_xxl
            from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer

            class WanT5EncoderWrapper:
                def __init__(self, cfg):
                    self.cfg = cfg
                    self.device = "cuda"
                    
                    print(f"Initializing Wan T5 Encoder with dtype bf16")
                    self.text_encoder = umt5_xxl(
                        encoder_only=True,
                        return_tokenizer=False,
                        dtype=torch.bfloat16,
                        device=torch.device("cpu")
                    ).eval().requires_grad_(False)
                    
                    if cfg.text_encoder.ckpt_path:
                         print(f"Loading T5 from {cfg.text_encoder.ckpt_path}")
                         sd = torch.load(cfg.text_encoder.ckpt_path, map_location="cpu", weights_only=True)
                         self.text_encoder.load_state_dict(sd)
                    else:
                         print("WARNING: No T5 ckpt_path provided in config! Using random weights if not using HF Hub.")
                    
                    self.text_encoder = self.text_encoder.to(self.device)
                    
                    self.tokenizer = HuggingfaceTokenizer(
                        name=cfg.text_encoder.name,
                        seq_len=cfg.text_encoder.text_len,
                        clean="whitespace",
                    )
                
                @torch.no_grad()
                def predict(self, prompts):
                    ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
                    ids = ids.to(self.device)
                    mask = mask.to(self.device)
                    seq_lens = mask.gt(0).sum(dim=1).long()
                    context = self.text_encoder(ids, mask)
                    return [u[:v].cpu() for u, v in zip(context, seq_lens)]
            
            t5_encoder = WanT5EncoderWrapper(self.root_cfg.algorithm)

        else:
            from algorithms.cogvideo.t5 import T5Encoder
            t5_encoder = T5Encoder(self.root_cfg.algorithm).cuda()

        dataset = self._build_dataset()
        records = dataset.records

        save_dir = self._get_save_dir(dataset)
        metadata_path = save_dir / dataset.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        print("Saving prompt embeddings and new metadata to ", save_dir)

        new_records = []
        for i in trange(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            prompts = [dataset.id_token + r["caption"] for r in batch]
            embeds = t5_encoder.predict(prompts)
            if hasattr(embeds, "cpu"):
                embeds = embeds.cpu()
            for r, embed in zip(batch, embeds):
                video_path = Path(r["video_path"])
                prompt_embed_path = (
                    save_dir / "prompt_embeds" / video_path.with_suffix(".pt")
                )
                prompt_embed_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(embed.clone(), prompt_embed_path)
                r["prompt_embed_path"] = str(prompt_embed_path.relative_to(save_dir))
                new_records.append(r)

        df = pd.DataFrame.from_records(new_records)
        df.to_csv(metadata_path, index=False)

        print("To review the prompt embeddings, go to ", save_dir)
        print(
            "If everything looks good, you can merge the new dataset into the old "
            "one with the following command:"
        )
        print(f"rsync -av {save_dir}/* {dataset.data_root} && rm -rf {save_dir}")

    def create_gemini_caption(self):
        """
        Create Gemini caption for each video in the dataset.

        1. Init the Dataset, and load all raw records.
        2. Init the GeminiCaptionProcessor with two params:  output_file and num_workers.
        3. Start the processor, and process each record. It will write to the output file.

        For each record in the dataset, it must has "video_path" as the absolute path.
        If each record has some additional keys, like: duration, fps, height, width, n_frames, youtube_key_segment, etc.
        they will be added to the output file. Check "Class VideoEntry" below for more details.

        Sample command:
        python main.py +name=create_gemini_caption experiment=process_data dataset=pandas experiment.tasks=[create_gemini_caption]
        """
        from utils.gemini_utils import GeminiCaptionProcessor

        # you need to install vertexai and setup google api to continue this.

        cfg = self.cfg.create_gemini_caption
        num_workers = cfg.n_workers

        dataset = self._build_dataset()
        records = dataset.records

        save_dir = self._get_save_dir(dataset)
        metadata_path = dataset.metadata_path.with_suffix(".json")
        metadata_path = metadata_path.parent / ("gemini_" + metadata_path.name)
        output_file = save_dir / metadata_path

        for r in records:
            r["video_path"] = str((dataset.data_root / r["video_path"]).absolute())

        if not os.path.exists(records[0]["video_path"]):
            raise ValueError("video_path must be an absolute path")

        processor = GeminiCaptionProcessor(output_file, num_workers=num_workers)
        processor.process_entries(records)
        print("To review the captions, go to ", output_file)
        print(
            "If everything looks good, you can merge the new dataset into the old "
            "one with the following command:"
        )
        print(f"rsync -av {save_dir}/* {dataset.data_root} && rm -rf {save_dir}")

    def run_hand_pose_estimation(self):

        import queue
        import threading
        import decord

        # see https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference/blob/main/image_pose_estimation.py
        from sapiens_inference import SapiensPoseEstimation, SapiensPoseEstimationType
        import time

        # also use confidence score > 0.3
        # for each key, it will store x, y, confidence score
        hand_keypoints_keys_list = [
            # in total of 40 keypoints
            # Right hand
            "right_wrist",
            "right_thumb4",
            "right_thumb3",
            "right_thumb2",
            "right_thumb_third_joint",
            "right_forefinger4",
            "right_forefinger3",
            "right_forefinger2",
            "right_forefinger_third_joint",
            "right_middle_finger4",
            "right_middle_finger3",
            "right_middle_finger2",
            "right_middle_finger_third_joint",
            "right_ring_finger4",
            "right_ring_finger3",
            "right_ring_finger2",
            "right_ring_finger_third_joint",
            "right_pinky_finger4",
            "right_pinky_finger3",
            "right_pinky_finger2",
            "right_pinky_finger_third_joint",
            # Left hand
            "left_wrist",
            "left_thumb4",
            "left_thumb3",
            "left_thumb2",
            "left_thumb_third_joint",
            "left_forefinger4",
            "left_forefinger3",
            "left_forefinger2",
            "left_forefinger_third_joint",
            "left_middle_finger4",
            "left_middle_finger3",
            "left_middle_finger2",
            "left_middle_finger_third_joint",
            "left_ring_finger4",
            "left_ring_finger3",
            "left_ring_finger2",
            "left_ring_finger_third_joint",
            "left_pinky_finger4",
            "left_pinky_finger3",
            "left_pinky_finger2",
            "left_pinky_finger_third_joint",
        ]

        cfg = self.cfg.run_hand_pose_estimation

        dataset = self._build_dataset()
        records = dataset.records

        # for debug, only process 50 videos
        # records = records[:50]
        # random sample 50 videos
        records = random.sample(records, 50)
        save_dir = self._get_save_dir(dataset)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving hand pose estimation results to {save_dir}")

        # Create queues for communication between producer and consumer
        frame_queue = queue.Queue(
            maxsize=100
        )  # Limit queue size to prevent memory issues
        STOP_TOKEN = "DONE"

        def producer(records, data_root):
            for record in records:
                try:
                    video_path = Path(data_root) / record["video_path"]
                    vr = decord.VideoReader(str(video_path))
                    n_frames = len(vr)

                    if n_frames == 0:
                        print(f"No frames found in {record['video_path']}")
                        continue

                    # Get first, middle, and last frame indices
                    frame_indices = [0, n_frames // 2, n_frames - 1]
                    frames = vr.get_batch(
                        frame_indices
                    ).asnumpy()  # Shape: (3, H, W, C)
                    # also resize each frame to 768x1024. with height 768, width 1024
                    # frames = [cv2.resize(frame, (1024, 768)) for frame in frames]
                    # Put frames and relative path in queue
                    frame_queue.put(
                        {
                            "frames": frames,
                            "video_path": str(
                                record["video_path"]
                            ),  # Keep relative path
                            "frame_indices": frame_indices,  # Keep track of which frames
                        }
                    )
                except Exception as e:
                    print(f"Error processing {record['video_path']}: {e}")
                    continue

            # Signal completion
            frame_queue.put(STOP_TOKEN)

        start_time = time.time()
        # Start producer thread
        producer_thread = threading.Thread(
            target=producer, args=(records, dataset.data_root), daemon=True
        )
        producer_thread.start()

        # Initialize the pose estimator
        dtype = torch.float16
        estimator = SapiensPoseEstimation(
            SapiensPoseEstimationType.POSE_ESTIMATION_03B, dtype=dtype
        )

        # Prepare a list to collect results
        # Each result will be a dict with video_path, frame_index, keypoints
        results = []

        while True:
            item = frame_queue.get()
            if item == STOP_TOKEN:
                break

            frames = item["frames"]  # Shape: (3, H, W, C)
            video_path = item["video_path"]
            frame_indices = item.get("frame_indices", [0, 1, 2])

            ret_per_video = {
                "video_path": video_path,
                "frame_indices": frame_indices,
                "keypoints_list": [],
            }
            for idx, frame in zip(frame_indices, frames):
                try:
                    # Convert frame from BGR (OpenCV) to RGB
                    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = frame

                    # Run pose estimation
                    result_img, keypoints = estimator(frame_rgb)

                    # Optionally, you can save or display the result_img
                    # For example, to save the annotated image:
                    # annotated_img_path = Path(save_dir) / f"{Path(video_path).stem}_frame_{idx}.jpg"
                    # cv2.imwrite(str(annotated_img_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

                    # Flatten keypoints and prepare the result entry
                    # Assuming keypoints is a NumPy array of shape (num_keypoints, 2) or similar
                    # print("debug", keypoints)
                    keypoints_flat = keypoints  # list of dict.

                    # only store the keypoints that are in hand_keypoints_keys_list
                    keypoints_flat = [
                        {
                            k: kp_dict[k]
                            for k in hand_keypoints_keys_list
                            if k in kp_dict
                        }
                        for kp_dict in keypoints_flat
                    ]

                    # then remove pred whose confidence score is less than 0.3
                    keypoints_flat = [
                        {k: v for k, v in kp_dict.items() if v[2] > 0.3}
                        for kp_dict in keypoints_flat
                    ]
                    result_entry = {
                        "frame_index": idx,
                        "keypoints_list": keypoints_flat,
                        "num_keypoints": sum([len(_) for _ in keypoints_flat]),
                    }

                    ret_per_video["keypoints_list"].append(result_entry)

                except Exception as e:
                    print(
                        f"Error running pose estimation for frame {idx} of {video_path}: {e}"
                    )
                    continue

            # tell if there exists any keypoints in the video, if not skip the video
            num_keypoints = sum(
                [_.get("num_keypoints", 0) for _ in ret_per_video["keypoints_list"]]
            )
            if num_keypoints > 0:
                results.append(ret_per_video)
            frame_queue.task_done()

        producer_thread.join()

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Total number of videos processed with keypoints: {len(results)}")

        # Convert results to JSON format
        if results:
            # Each result already contains:
            # - video_path
            # - frame_index
            # - keypoints_list (list of dictionaries with pose data)

            # Save to JSON
            json_path = Path(save_dir) / "hand_pose_results.json"
            import json

            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {json_path}")
        else:
            print("No results to save.")

    def run_human_detection(self):

        import queue
        import threading
        import decord
        from utils.detector_utils import Detector
        import time

        detector = Detector()  # bboxes = detector.detech(np_img_BGR)

        cfg = self.cfg.run_human_detection

        dataset = self._build_dataset()
        records = dataset.records
        # try 40k videos for now.
        # records = records[:40000]

        num_workers = cfg.total_workers
        job_id = cfg.job_id

        records = records[job_id::num_workers]
        # for debug, only process 50 videos
        # records = records[:50]
        # random sample 50 videos
        # records = random.sample(records, 50)
        save_dir = self._get_save_dir(dataset)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving hand pose estimation results to {save_dir}")

        # Create queues for communication between producer and consumer
        frame_queue = queue.Queue(
            maxsize=100
        )  # Limit queue size to prevent memory issues
        STOP_TOKEN = "DONE"

        def producer(records, data_root):
            for record in records:
                try:
                    video_path = Path(data_root) / record["video_path"]
                    vr = decord.VideoReader(str(video_path))
                    n_frames = len(vr)

                    if n_frames == 0:
                        print(f"No frames found in {record['video_path']}")
                        continue

                    # get one frame every second, read fps first then get frame indices
                    fps = vr.get_avg_fps()
                    frame_indices = [int(i * fps) for i in range(int(n_frames // fps))]
                    frames = vr.get_batch(
                        frame_indices
                    ).asnumpy()  # Shape: (n_f, H, W, C)
                    # also resize each frame to 768x1024. with height 768, width 1024
                    # frames = [cv2.resize(frame, (1024, 768)) for frame in frames]
                    # Put frames and relative path in queue
                    frame_queue.put(
                        {
                            "frames": frames,
                            "video_path": str(
                                record["video_path"]
                            ),  # Keep relative path
                            "frame_indices": frame_indices,  # Keep track of which frames
                        }
                    )
                except Exception as e:
                    print(f"Error processing {record['video_path']}: {e}")
                    continue

            # Signal completion
            frame_queue.put(STOP_TOKEN)

        start_time = time.time()
        # Start producer thread
        producer_thread = threading.Thread(
            target=producer, args=(records, dataset.data_root), daemon=True
        )
        producer_thread.start()

        # Initialize the pose estimator
        dtype = torch.float16

        # Prepare a list to collect results
        # Each result will be a dict with video_path, frame_index, keypoints
        results = []

        while True:
            item = frame_queue.get()
            if item == STOP_TOKEN:
                break

            frames = item["frames"]  # Shape: (3, H, W, C)
            video_path = item["video_path"]
            frame_indices = item.get("frame_indices", [0, 1, 2])

            ret_per_video = {
                "video_path": video_path,
                "frame_indices": frame_indices,
                "bbox_list": [],
            }
            num_detections = 0
            for idx, frame in zip(frame_indices, frames):
                try:
                    bboxes = detector.detect(
                        frame
                    ).tolist()  # [(x1, y1, x2, y2), ...] or empty list []
                    ret_per_video["bbox_list"].append(bboxes)
                    num_detections += len(bboxes)
                except Exception as e:
                    print(
                        f"Error running human detection for frame {idx} of {video_path}: {e}"
                    )
                    continue

            results.append(ret_per_video)
            frame_queue.task_done()

        producer_thread.join()

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Total number of videos processed with human detections: {len(results)}")

        # Convert results to JSON format
        if results:
            # Each result already contains:
            # - video_path
            # - frame_index
            # - bbox_list (list of list of bbox)

            # Save to JSON
            json_path = Path(save_dir) / f"human_detection_results_{job_id}.json"
            import json

            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {json_path}")
        else:
            print("No results to save.")
