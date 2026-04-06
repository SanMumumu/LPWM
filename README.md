# LPWM Training README

This README keeps only the training-related information for the current Flow-LPWM workflow in this fork.

## Scope

This fork currently trains only these 7 Flow-LPWM experiments:

- `Sketchy-U`
- `Sketchy-A`
- `BAIR-U`
- `Mario-U`
- `Bridge-L`
- `PandaPush`
- `OGBench-Scene`

## Environment

```bash
conda env create -f environment.yml
conda activate torch
```

Required training dependencies:

- `torch`
- `torchvision`
- `accelerate`
- `einops`
- `numpy`
- `scipy`
- `matplotlib`
- `opencv-python`
- `tqdm`
- `imageio`
- `ffmpeg`

## Project Root

All commands below assume:

```bash
cd /mnt/hwdata/wangsen/LPWM/LPWM
```

## Dataset Roots

The fixed training configs in this fork expect these dataset roots:

- `Sketchy`: `/mnt/hwdata/wangsen/LPWM/Data/sketchy_128/data`
- `BAIR`: `/mnt/hwdata/wangsen/LPWM/Data/bair_256/bair_256_ours`
- `Mario`: `/mnt/hwdata/wangsen/LPWM/Data/mario`
- `Bridge`: `/mnt/hwdata/wangsen/LPWM/Data/bridge_256/bridge`
- `PandaPush`: `/mnt/hwdata/wangsen/LPWM/Data/panda_ds/panda_ds`
- `OGBench-Scene`: `/mnt/hwdata/wangsen/LPWM/Data/ogbench_data/ogbench_ds`

## Training Config Files

The training configs used in this fork are:

- `/configs/sketchy_u_flow_exp.json`
- `/configs/sketchy_a_flow_exp.json`
- `/configs/bair_u_flow_exp.json`
- `/configs/mario_u_flow_exp.json`
- `/configs/bridge_l_flow_exp.json`
- `/configs/pandapush_flow_exp.json`
- `/configs/ogbench_scene_flow_exp.json`

These configs already contain the dataset paths, `model_name="flow_lpwm"`, and the dataset-specific epoch counts.

## Epochs

The current `num_epochs` values are:

- `Sketchy-U`: `18`
- `Sketchy-A`: `18`
- `BAIR-U`: `20`
- `Mario-U`: `200`
- `Bridge-L`: `42`
- `PandaPush`: `84`
- `OGBench-Scene`: `80`

## 8-GPU Training

This fork is set up to train sequentially on 8 GPUs by default through [exp_todo.sh](/mnt/hwdata/wangsen/LPWM/LPWM/exp_todo.sh).

Standard command:

```bash
cd /mnt/hwdata/wangsen/LPWM/LPWM
conda activate torch
RUN_TRAIN=1 bash ./exp_todo.sh
```

Explicitly pin the 8 GPUs:

```bash
cd /mnt/hwdata/wangsen/LPWM/LPWM
conda activate torch
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 RUN_TRAIN=1 bash ./exp_todo.sh
```

What this does:

- Runs the 7 experiments one-by-one
- Uses `accelerate` with `./accel_conf.yml`
- Uses `train_lpwm_accelerate.py`
- Uses the fixed config files listed above

## Single Experiment Training

If you want to train only one experiment, run its config directly.

Examples:

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/sketchy_u_flow_exp.json
```

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/sketchy_a_flow_exp.json
```

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/bair_u_flow_exp.json
```

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/mario_u_flow_exp.json
```

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/bridge_l_flow_exp.json
```

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/pandapush_flow_exp.json
```

```bash
LPWM_ACCELERATE_GPUS=0,1,2,3,4,5,6,7 accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d ./configs/ogbench_scene_flow_exp.json
```

## Single Experiment Evaluating

After training, replace `RUN_DIR` with the actual experiment directory created by training.

Example:

```bash
RUN_DIR=./300330_031419_sketchy_flowlpwm_flow_sketchy_u_exp
```

Use the best LPIPS checkpoint stored under `RUN_DIR/saves/`.

The `t` value below is the dataset training horizon from the paper. The evaluation commands use:

- `-c`: conditioning frames
- `--horizon`: final video length

For LPWM evaluation, `--horizon` is the total clip length used by the script, not the future prediction length `p`.
The generated future length is `--horizon - -c`.

### Sketchy-U

Reference setting: `t=20, c=6, p=44`, so use `--horizon 50`

```bash
RUN_DIR=./<your_sketchy_u_run_dir>
python eval/eval_gen_metrics.py -d sketchy -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/sketchy_flowlpwm_flow_sketchy_u_exp_best_lpips.pth" --sample -b 10 -c 6 --horizon 50 --prefix "" --ctx
python eval/eval_fvd.py -d sketchy -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/sketchy_flowlpwm_flow_sketchy_u_exp_best_lpips.pth" --sample -b 4 -c 6 --horizon 50 --prefix "" --n_videos_per_clip 1
python generate_lpwm_video_prediction.py -d sketchy -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/sketchy_flowlpwm_flow_sketchy_u_exp_best_lpips.pth" --sample -n 4 -c 6 --horizon 50 --prefix ""
```

### Sketchy-A

Reference setting: `t=20, c=6, p=44`, so use `--horizon 50`

```bash
RUN_DIR=./<your_sketchy_a_run_dir>
python eval/eval_gen_metrics.py -d sketchy -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/sketchy_flowlpwm_flow_sketchy_a_exp_best_lpips.pth" --sample -b 10 -c 6 --horizon 50 --prefix "" --ctx
python eval/eval_fvd.py -d sketchy -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/sketchy_flowlpwm_flow_sketchy_a_exp_best_lpips.pth" --sample -b 4 -c 6 --horizon 50 --prefix "" --n_videos_per_clip 1
python generate_lpwm_video_prediction.py -d sketchy -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/sketchy_flowlpwm_flow_sketchy_a_exp_best_lpips.pth" --sample -n 4 -c 6 --horizon 50 --prefix ""
```

### BAIR-U

Reference setting: `t=16, c=1, p=15`, so use `--horizon 16`

```bash
RUN_DIR=./<your_bair_u_run_dir>
python eval/eval_gen_metrics.py -d bair -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/bair_flowlpwm_flow_bair_u_exp_best_lpips.pth" --sample -b 10 -c 1 --horizon 16 --prefix "" --ctx
python eval/eval_fvd.py -d bair -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/bair_flowlpwm_flow_bair_u_exp_best_lpips.pth" --sample -b 4 -c 1 --horizon 16 --prefix "" --n_videos_per_clip 1
python generate_lpwm_video_prediction.py -d bair -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/bair_flowlpwm_flow_bair_u_exp_best_lpips.pth" --sample -n 4 -c 1 --horizon 16 --prefix ""
```

### Mario-U

Reference setting: `t=20, c=6, p=34`, so use `--horizon 40`

```bash
RUN_DIR=./<your_mario_u_run_dir>
python eval/eval_gen_metrics.py -d mario -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/mario_flowlpwm_flow_mario_u_exp_best_lpips.pth" --sample -b 10 -c 6 --horizon 40 --prefix "" --ctx
python eval/eval_fvd.py -d mario -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/mario_flowlpwm_flow_mario_u_exp_best_lpips.pth" --sample -b 4 -c 6 --horizon 40 --prefix "" --n_videos_per_clip 1
python generate_lpwm_video_prediction.py -d mario -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/mario_flowlpwm_flow_mario_u_exp_best_lpips.pth" --sample -n 4 -c 6 --horizon 40 --prefix ""
```

### Bridge-L

Reference setting: `t=24, c=1, p=29`, so use `--horizon 30`

```bash
RUN_DIR=./<your_bridge_l_run_dir>
python eval/eval_gen_metrics.py -d bridge -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/bridge_flowlpwm_flow_bridge_l_exp_best_lpips.pth" --sample -b 10 -c 1 --horizon 30 --prefix "" --ctx
python eval/eval_fvd.py -d bridge -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/bridge_flowlpwm_flow_bridge_l_exp_best_lpips.pth" --sample -b 4 -c 1 --horizon 30 --prefix "" --n_videos_per_clip 1
python generate_lpwm_video_prediction.py -d bridge -p "${RUN_DIR}" --checkpoint "${RUN_DIR}/saves/bridge_flowlpwm_flow_bridge_l_exp_best_lpips.pth" --sample -n 4 -c 1 --horizon 30 --prefix ""
```

## Training Entry Points

Files used for training in this fork:

- [exp_todo.sh](D:/Gitdesktop/LPWM/exp_todo.sh)
- [train_lpwm.py](D:/Gitdesktop/LPWM/train_lpwm.py)
- [train_lpwm_accelerate.py](D:/Gitdesktop/LPWM/train_lpwm_accelerate.py)
- [accel_conf.yml](D:/Gitdesktop/LPWM/accel_conf.yml)

## Notes

- `exp_todo.sh` is the recommended entry point for the current project.
- `train_lpwm_accelerate.py` reads `LPWM_ACCELERATE_GPUS` and sets `CUDA_VISIBLE_DEVICES` from it.
- `OGBench-Scene` is restricted to the `scene` subset in `/configs/ogbench_scene_flow_exp.json`.
- `PandaPush` currently uses the explicit `dataset_kwargs.tasks=["C","T"]` setting stored in `/configs/pandapush_flow_exp.json`.
