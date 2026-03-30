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
