import torch

from flow_lpwm import FlowLPWM
from models import DLP
from utils.util_func import get_config


MODEL_TAGS = {
    "dlp": "gddlp",
    "flow_lpwm": "flowlpwm",
}


def get_model_name(config):
    return str(config.get("model_name", "dlp")).lower()


def get_model_tag(config):
    model_name = get_model_name(config)
    if model_name not in MODEL_TAGS:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return MODEL_TAGS[model_name]


def build_model_from_config(config):
    model_name = get_model_name(config)

    common_kwargs = dict(
        cdim=config["ch"],
        image_size=config["image_size"],
        normalize_rgb=config["normalize_rgb"],
        n_views=config.get("n_views", 1),
        n_kp_per_patch=config["n_kp_per_patch"],
        patch_size=config["patch_size"],
        anchor_s=config["anchor_s"],
        n_kp_enc=config["n_kp_enc"],
        n_kp_prior=config["n_kp_prior"],
        warmup_n_kp_ratio=config.get("warmup_n_kp_ratio", 1.0),
        mask_bg_in_enc=config.get("mask_bg_in_enc", True),
        pad_mode=config["pad_mode"],
        dropout=config["dropout"],
        features_dist=config.get("features_dist", "gauss"),
        learned_feature_dim=config["learned_feature_dim"],
        learned_bg_feature_dim=config.get("learned_bg_feature_dim", config["learned_feature_dim"]),
        n_fg_categories=config.get("n_fg_categories", 8),
        n_fg_classes=config.get("n_fg_classes", 4),
        n_bg_categories=config.get("n_bg_categories", 4),
        n_bg_classes=config.get("n_bg_classes", 4),
        scale_std=config["scale_std"],
        offset_std=config["offset_std"],
        obj_on_alpha=config["obj_on_alpha"],
        obj_on_beta=config["obj_on_beta"],
        obj_on_min=config.get("obj_on_min", 1e-4),
        obj_on_max=config.get("obj_on_max", 100),
        obj_res_from_fc=config["obj_res_from_fc"],
        obj_ch_mult_prior=config.get("obj_ch_mult_prior", config["obj_ch_mult"]),
        obj_ch_mult=config["obj_ch_mult"],
        obj_base_ch=config["obj_base_ch"],
        obj_final_cnn_ch=config["obj_final_cnn_ch"],
        bg_res_from_fc=config["bg_res_from_fc"],
        bg_ch_mult=config["bg_ch_mult"],
        bg_base_ch=config["bg_base_ch"],
        bg_final_cnn_ch=config["bg_final_cnn_ch"],
        use_resblock=config["use_resblock"],
        num_res_blocks=config["num_res_blocks"],
        cnn_mid_blocks=config.get("cnn_mid_blocks", False),
        mlp_hidden_dim=config.get("mlp_hidden_dim", 256),
        attn_norm_type=config.get("attn_norm_type", "rms"),
        embed_init_std=config.get("embed_init_std", 0.02),
        particle_positional_embed=config.get("particle_positional_embed", True),
        use_z_orig=config.get("use_z_orig", True),
        particle_score=config.get("particle_score", False),
        filtering_heuristic=config.get("filtering_heuristic", "none"),
        timestep_horizon=config["timestep_horizon"],
        n_static_frames=config["num_static_frames"],
        predict_delta=config["predict_delta"],
        context_dim=config["context_dim"],
        ctx_dist=config.get("context_dist", "gauss"),
        n_ctx_categories=config.get("n_ctx_categories", 8),
        n_ctx_classes=config.get("n_ctx_classes", 4),
        causal_ctx=config.get("causal_ctx", True),
        ctx_pool_mode=config.get("ctx_pool_mode", "none"),
        global_ctx_pool=config.get("global_ctx_pool", False),
        pool_ctx_dim=config.get("pool_ctx_dim", config.get("context_dim", 7)),
        n_pool_ctx_categories=config.get("n_pool_ctx_categories", 8),
        n_pool_ctx_classes=config.get("n_pool_ctx_classes", 4),
        global_local_fuse_mode=config.get("global_local_fuse_mode", "none"),
        condition_local_on_global=config.get("condition_local_on_global", True),
        pint_dyn_layers=config["pint_dyn_layers"],
        pint_dyn_heads=config["pint_dyn_heads"],
        pint_dim=config["pint_dim"],
        pint_ctx_layers=config["pint_ctx_layers"],
        pint_ctx_heads=config["pint_ctx_heads"],
        pint_enc_layers=config["pint_enc_layers"],
        pint_enc_heads=config["pint_enc_heads"],
        action_condition=config.get("action_condition", False),
        action_dim=config.get("action_dim", 0),
        null_action_embed=config.get("null_action_embed", False),
        random_action_condition=config.get("random_action_condition", False),
        random_action_dim=config.get("random_action_dim", 0),
        action_in_ctx_module=config.get("action_in_ctx_module", True),
        language_condition=config.get("language_condition", False),
        language_embed_dim=config.get("language_embed_dim", 0),
        language_max_len=config.get("language_max_len", 32),
        img_goal_condition=config.get("image_goal_condition", False),
    )

    if model_name == "dlp":
        return DLP(**common_kwargs)

    if model_name == "flow_lpwm":
        return FlowLPWM(
            **common_kwargs,
            beta_flow=config.get("beta_flow", 1.0),
            flow_sample_steps=config.get("flow_sample_steps", 10),
            router_threshold=config.get("router_threshold", 0.05),
            router_min_keep=config.get("router_min_keep", 1),
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_from_config(conf_path, ckpt_path=None, model_name_override=None):
    config = get_config(conf_path)
    if model_name_override is not None:
        config = dict(config)
        config["model_name"] = model_name_override

    model = build_model_from_config(config)
    if ckpt_path is not None:
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False))
            print("loaded model from checkpoint")
        except Exception:
            print("model checkpoint not found")
    return model, config
