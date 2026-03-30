import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DLP
from modules.modules import ParticleContextDecoder, RMSNorm


class SparseParticleRouter(nn.Module):
    def __init__(self, threshold=0.05, min_keep=1):
        super().__init__()
        self.threshold = threshold
        self.min_keep = max(int(min_keep), 1)

    def _reshape_obj_on(self, z_obj_on, n_views=1):
        if z_obj_on is None:
            return None
        if len(z_obj_on.shape) == 3:
            z_obj_on = z_obj_on.unsqueeze(-1)
        if n_views > 1:
            z_obj_on = z_obj_on.view(-1, n_views, *z_obj_on.shape[1:])
            z_obj_on = z_obj_on.permute(0, 2, 1, 3, 4).reshape(
                z_obj_on.shape[0], z_obj_on.shape[2], -1, z_obj_on.shape[-1]
            )
        return z_obj_on

    def route_tokens(self, tokens, z_obj_on, context_tokens=None, n_views=1, keep_bg=True):
        if z_obj_on is None:
            particle_pad_mask = torch.ones(tokens.shape[0], tokens.shape[2], dtype=torch.bool, device=tokens.device)
            return tokens, context_tokens, particle_pad_mask, None

        z_obj_on = self._reshape_obj_on(z_obj_on, n_views=n_views)
        batch_size, _, fg_particles, _ = z_obj_on.shape
        total_tokens = tokens.shape[2]
        bg_tokens = min(n_views, max(total_tokens - fg_particles, 0)) if keep_bg else 0
        extra_tokens = max(total_tokens - fg_particles - bg_tokens, 0)

        scores = z_obj_on.squeeze(-1).amax(dim=1)
        active_mask = scores > self.threshold

        topk = min(self.min_keep, fg_particles)
        fallback_idx = scores.topk(topk, dim=-1).indices
        fallback_mask = torch.zeros_like(active_mask)
        fallback_mask.scatter_(1, fallback_idx, True)
        has_any = active_mask.any(dim=-1, keepdim=True)
        active_mask = torch.where(has_any, active_mask, fallback_mask)

        active_counts = active_mask.sum(dim=-1)
        max_active = max(int(active_counts.max().item()), 1)

        gathered_tokens = []
        gathered_context = [] if context_tokens is not None else None
        gathered_indices = []
        fg_valid_masks = []
        active_means = []

        for batch_idx in range(batch_size):
            indices = torch.nonzero(active_mask[batch_idx], as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                indices = scores[batch_idx].topk(topk, dim=-1).indices
            indices = indices.sort().values
            valid_count = indices.numel()
            active_means.append(float(valid_count))

            padded_indices = torch.zeros(max_active, dtype=torch.long, device=tokens.device)
            padded_indices[:valid_count] = indices
            valid_mask = torch.zeros(max_active, dtype=torch.bool, device=tokens.device)
            valid_mask[:valid_count] = True

            fg_tokens = tokens[batch_idx, :, :fg_particles]
            selected_fg = fg_tokens[:, padded_indices]

            parts = [selected_fg]
            if bg_tokens:
                parts.append(tokens[batch_idx, :, fg_particles : fg_particles + bg_tokens])
            if extra_tokens:
                parts.append(tokens[batch_idx, :, fg_particles + bg_tokens :])
            gathered_tokens.append(torch.cat(parts, dim=1))

            if context_tokens is not None:
                fg_context = context_tokens[batch_idx, :, :fg_particles]
                selected_context = fg_context[:, padded_indices]
                context_parts = [selected_context]
                if bg_tokens:
                    context_parts.append(context_tokens[batch_idx, :, fg_particles : fg_particles + bg_tokens])
                if extra_tokens:
                    context_parts.append(context_tokens[batch_idx, :, fg_particles + bg_tokens :])
                gathered_context.append(torch.cat(context_parts, dim=1))

            gathered_indices.append(padded_indices)
            fg_valid_masks.append(valid_mask)

        sparse_tokens = torch.stack(gathered_tokens, dim=0)
        sparse_context = torch.stack(gathered_context, dim=0) if gathered_context is not None else None
        gathered_indices = torch.stack(gathered_indices, dim=0)
        fg_valid_masks = torch.stack(fg_valid_masks, dim=0)
        tail_valid = torch.ones(batch_size, bg_tokens + extra_tokens, dtype=torch.bool, device=tokens.device)
        particle_pad_mask = torch.cat([fg_valid_masks, tail_valid], dim=1)
        route_state = {
            "fg_particles": fg_particles,
            "bg_tokens": bg_tokens,
            "extra_tokens": extra_tokens,
            "indices": gathered_indices,
            "fg_valid_masks": fg_valid_masks,
            "active_particles_mean": tokens.new_tensor(active_means).mean(),
        }
        return sparse_tokens, sparse_context, particle_pad_mask, route_state

    def scatter_tokens(self, sparse_tokens, route_state, dense_tokens):
        if route_state is None:
            return sparse_tokens

        dense_out = dense_tokens.clone()
        fg_particles = route_state["fg_particles"]
        indices = route_state["indices"]
        fg_valid_masks = route_state["fg_valid_masks"]
        bg_tokens = route_state["bg_tokens"]
        extra_tokens = route_state["extra_tokens"]
        max_active = fg_valid_masks.shape[1]

        for batch_idx in range(dense_out.shape[0]):
            valid_indices = indices[batch_idx][fg_valid_masks[batch_idx]]
            valid_sparse = sparse_tokens[batch_idx, :, :max_active][:, fg_valid_masks[batch_idx]]
            dense_out[batch_idx, :, valid_indices] = valid_sparse
            cursor = max_active
            if bg_tokens:
                dense_out[batch_idx, :, fg_particles : fg_particles + bg_tokens] = sparse_tokens[
                    batch_idx, :, cursor : cursor + bg_tokens
                ]
                cursor += bg_tokens
            if extra_tokens:
                dense_out[batch_idx, :, fg_particles + bg_tokens :] = sparse_tokens[batch_idx, :, cursor:]

        return dense_out


class FlowContextModule(nn.Module):
    def __init__(self, backbone, flow_sample_steps=10):
        super().__init__()
        self.backbone = backbone
        self.context_dim = backbone.context_dim
        self.hidden_dim = backbone.hidden_dim
        self.flow_sample_steps = max(int(flow_sample_steps), 1)
        self.inverse_decoder = ParticleContextDecoder(
            n_particles=backbone.n_kp_enc,
            input_dim=backbone.projection_dim,
            hidden_dim=backbone.hidden_dim,
            context_dist="gauss",
            context_dim=backbone.context_dim,
            n_ctx_categories=backbone.n_ctx_categories,
            n_ctx_classes=backbone.n_ctx_classes,
            learned_ctx_token=backbone.learned_ctx_token,
            ctx_pool_mode=backbone.ctx_pool_mode,
            shared_logvar=False,
            output_ctx_logvar=False,
            conditional=False,
            cond_dim=0,
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(backbone.projection_dim, backbone.hidden_dim),
            RMSNorm(backbone.hidden_dim),
            nn.GELU(),
        )
        self.vector_field_net = nn.Sequential(
            nn.Linear(backbone.hidden_dim + backbone.context_dim + 1, backbone.hidden_dim),
            RMSNorm(backbone.hidden_dim),
            nn.GELU(),
            nn.Linear(backbone.hidden_dim, backbone.hidden_dim),
            nn.GELU(),
            nn.Linear(backbone.hidden_dim, backbone.context_dim),
        )
        self.last_flow_loss = None

    def _get_condition_tokens(
        self,
        z,
        z_scale,
        z_obj_on,
        z_depth,
        z_features,
        z_bg_features=None,
        z_base_var=None,
        z_score=None,
        patch_id_embed=None,
        deterministic=False,
        warmup=False,
        actions=None,
        actions_mask=None,
        lang_embed=None,
        z_goal=None,
    ):
        return self.backbone(
            z,
            z_scale,
            z_obj_on,
            z_depth,
            z_features,
            z_bg_features,
            z_base_var,
            z_score,
            patch_id_embed,
            deterministic=deterministic,
            warmup=warmup,
            encode_posterior=False,
            encode_prior=False,
            actions=actions,
            actions_mask=actions_mask,
            lang_embed=lang_embed,
            z_goal=z_goal,
        )

    def compute_flow_loss(self, condition_tokens, z_context_target):
        cond = self.condition_proj(condition_tokens)
        time = torch.rand(*z_context_target.shape[:-1], 1, device=z_context_target.device)
        z_noise = torch.randn_like(z_context_target)
        z_interp = (1.0 - time) * z_noise + time * z_context_target
        target_velocity = z_context_target - z_noise
        pred_velocity = self.vector_field_net(torch.cat([cond, z_interp, time], dim=-1))
        return F.mse_loss(pred_velocity, target_velocity)

    def sample_policy(self, condition_tokens, steps=None):
        steps = max(int(steps or self.flow_sample_steps), 1)
        cond = self.condition_proj(condition_tokens)
        sample = torch.randn(
            *condition_tokens.shape[:-1],
            self.context_dim,
            device=condition_tokens.device,
            dtype=condition_tokens.dtype,
        )
        dt = 1.0 / steps
        for step in range(steps):
            t_value = step * dt
            t_tensor = torch.full((*sample.shape[:-1], 1), t_value, device=sample.device, dtype=sample.dtype)
            sample = sample + self.vector_field_net(torch.cat([cond, sample, t_tensor], dim=-1)) * dt
        return sample

    def forward(
        self,
        z,
        z_scale,
        z_obj_on,
        z_depth,
        z_features,
        z_bg_features=None,
        z_base_var=None,
        z_score=None,
        patch_id_embed=None,
        deterministic=False,
        warmup=False,
        encode_posterior=True,
        encode_prior=True,
        actions=None,
        actions_mask=None,
        lang_embed=None,
        z_goal=None,
    ):
        backbone_out = self._get_condition_tokens(
            z,
            z_scale,
            z_obj_on,
            z_depth,
            z_features,
            z_bg_features=z_bg_features,
            z_base_var=z_base_var,
            z_score=z_score,
            patch_id_embed=patch_id_embed,
            deterministic=deterministic,
            warmup=warmup,
            actions=actions,
            actions_mask=actions_mask,
            lang_embed=lang_embed,
            z_goal=z_goal,
        )
        condition_tokens = backbone_out["ctx_backbone_tokens"]
        self.last_flow_loss = None

        if encode_posterior:
            inverse_out = self.inverse_decoder(condition_tokens, deterministic=True)
            mu_context = inverse_out["mu_context"]
            z_context = inverse_out["z_context"]
            logvar_context = torch.zeros_like(mu_context)
            self.last_flow_loss = self.compute_flow_loss(condition_tokens, z_context)
        else:
            mu_context = logvar_context = z_context = None

        if encode_prior:
            z_context_dyn = self.sample_policy(condition_tokens)
            mu_context_dyn = z_context_dyn
            logvar_context_dyn = torch.zeros_like(z_context_dyn)
        else:
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None

        return {
            "mu_context": mu_context,
            "logvar_context": logvar_context,
            "z_context": z_context,
            "mu_context_dyn": mu_context_dyn,
            "logvar_context_dyn": logvar_context_dyn,
            "z_context_dyn": z_context_dyn,
            "mu_context_global": None,
            "logvar_context_global": None,
            "z_context_global": None,
            "mu_context_global_dyn": None,
            "logvar_context_global_dyn": None,
            "z_context_global_dyn": None,
            "loss_flow_context": self.last_flow_loss,
            "ctx_backbone_tokens": condition_tokens,
            "ctx_particle_pad_mask": backbone_out.get("ctx_particle_pad_mask"),
            "z_goal_proj": backbone_out["z_goal_proj"],
        }


class FlowLPWM(DLP):
    def __init__(self, beta_flow=1.0, flow_sample_steps=10, router_threshold=0.05, router_min_keep=1, **kwargs):
        super().__init__(**kwargs)
        if self.ctx_pool_mode != "none":
            raise ValueError("FlowLPWM v1 only supports ctx_pool_mode='none'")
        if self.global_ctx_pool:
            raise ValueError("FlowLPWM v1 does not support global_ctx_pool")
        if self.context_dist != "gauss":
            raise ValueError("FlowLPWM v1 expects context_dist='gauss'")

        self.beta_flow = beta_flow
        self.model_name = "flow_lpwm"
        self.particle_router = SparseParticleRouter(threshold=router_threshold, min_keep=router_min_keep)

        old_ctx_module = self.ctx_module
        old_ctx_module.particle_router = self.particle_router
        self.ctx_module = FlowContextModule(old_ctx_module, flow_sample_steps=flow_sample_steps)
        self.encoder_module.ctx_enc = self.ctx_module
        self.encoder_module.use_ctx_enc = True

        self.dyn_module.context_decoder = self.ctx_module
        self.dyn_module.particle_router = self.particle_router

    def calc_dyn_elbo(
        self,
        x,
        model_output,
        warmup=False,
        beta_kl=0.1,
        beta_dyn=0.1,
        beta_rec=1.0,
        kl_balance=0.001,
        dynamic_discount=None,
        recon_loss_type="mse",
        recon_loss_func=None,
        balance=0.5,
        beta_dyn_rec=1.0,
        num_static=1,
        use_kl_mask=True,
        apply_mask_on_obj_on=False,
        beta_obj=0.0,
        done_mask=None,
    ):
        loss_dict = super().calc_dyn_elbo(
            x,
            model_output,
            warmup=warmup,
            beta_kl=beta_kl,
            beta_dyn=beta_dyn,
            beta_rec=beta_rec,
            kl_balance=kl_balance,
            dynamic_discount=dynamic_discount,
            recon_loss_type=recon_loss_type,
            recon_loss_func=recon_loss_func,
            balance=balance,
            beta_dyn_rec=beta_dyn_rec,
            num_static=num_static,
            use_kl_mask=use_kl_mask,
            apply_mask_on_obj_on=apply_mask_on_obj_on,
            beta_obj=beta_obj,
            done_mask=done_mask,
        )

        flow_loss = self.ctx_module.last_flow_loss
        if flow_loss is None:
            flow_loss = torch.tensor(0.0, device=x.device)
        active_particles_mean = getattr(self.dyn_module, "last_active_particles_mean", None)
        if active_particles_mean is None:
            active_particles_mean = torch.tensor(0.0, device=x.device)

        loss_scale = 0.1 if recon_loss_type == "mse" else 0.01
        norm_f = 1.0 / (self.timestep_horizon + 1) if (done_mask is None or warmup) else 1.0
        loss_dict["loss"] = (
            loss_dict["loss"]
            - loss_scale * norm_f * beta_dyn * loss_dict["loss_kl_context"]
            + loss_scale * norm_f * self.beta_flow * flow_loss
        )
        loss_dict["loss_flow_context"] = flow_loss
        loss_dict["loss_kl_context"] = flow_loss
        loss_dict["active_particles_mean"] = active_particles_mean.detach()
        return loss_dict
