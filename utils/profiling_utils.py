from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Optional, Any, Union, Callable, Tuple, Iterable, Iterator

import os
import torch
import torch.nn as nn
from transformers import TrainerCallback, PreTrainedModel
from trl import DPOTrainer, SFTTrainer
from torch.func import functional_call, jvp, vmap
from torch.nn.attention import SDPBackend, sdpa_kernel


IGNORE_INDEX = -100
LOSS_SPACE_TOKEN_CHUNK_SIZE = 32


def _is_trainable(model: nn.Module) -> bool:
    return next(iter(model.parameters())).requires_grad


def supervised_position_mask_from_labels(
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise RuntimeError(f"Expected labels with shape [batch, seq], got {tuple(labels.shape)}")

    position_mask = torch.zeros_like(labels, dtype=torch.bool)
    position_mask[:, :-1] = labels[:, 1:].ne(ignore_index)
    position_mask[:, -1] = False
    return position_mask


def next_token_targets_from_position_mask(
    token_ids: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    if token_ids.shape != position_mask.shape:
        raise RuntimeError(
            f"Expected token ids with shape {tuple(position_mask.shape)}, got {tuple(token_ids.shape)}"
        )
    if position_mask[:, -1].any():
        raise RuntimeError("The last position has no next token and must not be selected.")

    return token_ids[:, 1:][position_mask[:, :-1]]


@contextmanager
def _gradient_checkpointing_disabled(model: nn.Module):
    previous = [
        (module, module.gradient_checkpointing)
        for module in model.modules()
        if getattr(module, "gradient_checkpointing", False)
    ]

    for module, _ in previous:
        module.gradient_checkpointing = False
    try:
        yield
    finally:
        for module, flag in previous:
            module.gradient_checkpointing = flag


@dataclass
class FeatureParameterCollector:
    current_update_step: int = 0
    handles: list = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    model_params: Optional[list[dict[str, Any]]] = None
    first_layer_inputs: Optional[dict[str, torch.Tensor | Tuple[torch.Tensor]]] = None
    assistant_only_mask: Optional[torch.Tensor] = None
    enabled: bool = False
    profile_batch_size: Optional[int] = None
    batch_index: Optional[torch.Tensor] = None
    store_gradients: bool = False
    gradients: Optional[list[dict[str, torch.Tensor]]] = None

    def begin_update(self, update_step: int, model: PreTrainedModel) -> None:
        self.current_update_step = update_step
        self.records.clear()
        self.model_params = []
        self.gradients = None

        for layer in model.model.layers:
            if not _is_trainable(layer):
                self.model_params.append({})
            else:
                self.model_params.append(
                    {
                        name: param.detach().clone()
                        for name, param in layer.named_parameters()
                    }
                )

    @torch.no_grad()
    def capture_gradients(self, model: PreTrainedModel) -> None:
        if not self.store_gradients:
            return

        self.gradients = []
        for layer in model.model.layers:
            if not _is_trainable(layer):
                self.gradients.append({})
            else:
                self.gradients.append(
                    {
                        name: param.grad.detach().clone()
                        for name, param in layer.named_parameters()
                        if param.grad is not None
                    }
                )

    def begin_forward(self, inputs: dict[str, torch.Tensor | Any]) -> None:
        if 'labels' not in inputs:
            raise ValueError("Expected `labels` in the training inputs.")

        mask = supervised_position_mask_from_labels(inputs['labels'].detach())
        if self.profile_batch_size is not None and self.profile_batch_size < mask.shape[0]:
            self.batch_index = torch.arange(self.profile_batch_size, device=mask.device)
            mask = mask[self.batch_index]
        else:
            self.batch_index = None

        self.enabled = True
        self.assistant_only_mask = mask

    def _select(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.batch_index is None or tensor.shape[0] == 1:
            return tensor
        return tensor[self.batch_index]

    def make_forward_hook(self):
        def hook_fn(module, args, kwargs):
            if not self.enabled:
                return

            self.first_layer_inputs = {}
            self.first_layer_inputs['hidden_states'] = self._select(args[0]).detach().clone()

            for name in ("attention_mask", "position_embeddings", "position_ids"):
                value = kwargs.get(name)

                if torch.is_tensor(value):
                    self.first_layer_inputs[name] = self._select(value).detach().clone()
                elif isinstance(value, (tuple, list)):
                    self.first_layer_inputs[name] = tuple(
                        self._select(x).detach().clone() if torch.is_tensor(x) else x
                        for x in value
                    )
        return hook_fn

    def end_forward(self):
        self.records.append({
            'first_layer_inputs': self.first_layer_inputs,
            'assistant_only_mask': self.assistant_only_mask,
        })

    def reset_forward(self):
        self.first_layer_inputs = None
        self.assistant_only_mask = None
        self.enabled = False

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@dataclass
class FeatureParameterDPOCollector(FeatureParameterCollector):

    def begin_forward(self, inputs: dict[str, torch.Tensor | Any]) -> None:
        mask = supervised_position_mask_from_labels(
            inputs['completion_mask'].detach(), ignore_index=0,
        )
        if self.profile_batch_size is not None and self.profile_batch_size < mask.shape[0]:
            half = torch.arange(self.profile_batch_size // 2, device=mask.device)
            self.batch_index = torch.cat([half, mask.shape[0] // 2 + half])
            mask = mask[self.batch_index]
        else:
            self.batch_index = None

        self.enabled = True
        self.assistant_only_mask = mask


@dataclass
class LossSpaceCollector(FeatureParameterCollector):
    store_gradients: bool = True
    labels: Optional[torch.Tensor] = None

    def begin_forward(self, inputs: dict[str, torch.Tensor | Any]) -> None:
        super().begin_forward(inputs)
        self.labels = self._select(inputs['labels'].detach().clone())

    def end_forward(self):
        self.records.append({
            'first_layer_inputs': self.first_layer_inputs,
            'assistant_only_mask': self.assistant_only_mask,
            'targets': next_token_targets_from_position_mask(self.labels, self.assistant_only_mask),
        })

    def reset_forward(self):
        super().reset_forward()
        self.labels = None


@dataclass
class LossSpaceDPOCollector(FeatureParameterDPOCollector):
    store_gradients: bool = True
    input_ids: Optional[torch.Tensor] = None
    ref_chosen_logps: Optional[torch.Tensor] = None
    ref_rejected_logps: Optional[torch.Tensor] = None

    def begin_forward(self, inputs: dict[str, torch.Tensor | Any]) -> None:
        if 'ref_chosen_logps' not in inputs or 'ref_rejected_logps' not in inputs:
            raise ValueError("Expected precomputed reference log-probs in the training inputs.")

        super().begin_forward(inputs)
        num_pairs = self.assistant_only_mask.shape[0] // 2

        self.input_ids = self._select(inputs['input_ids'].detach().clone())
        self.ref_chosen_logps = inputs['ref_chosen_logps'].detach()[:num_pairs].float().clone()
        self.ref_rejected_logps = inputs['ref_rejected_logps'].detach()[:num_pairs].float().clone()

    def end_forward(self):
        self.records.append({
            'first_layer_inputs': self.first_layer_inputs,
            'assistant_only_mask': self.assistant_only_mask,
            'targets': next_token_targets_from_position_mask(self.input_ids, self.assistant_only_mask),
            'ref_chosen_logps': self.ref_chosen_logps,
            'ref_rejected_logps': self.ref_rejected_logps,
        })

    def reset_forward(self):
        super().reset_forward()
        self.input_ids = None
        self.ref_chosen_logps = None
        self.ref_rejected_logps = None


class FeatureParameterProfilingMixin:
    def __init__(self, *trainer_args, collector: FeatureParameterCollector, **trainer_kwargs):
        super().__init__(*trainer_args, **trainer_kwargs)
        self.feat_param_collector = collector

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        self.feat_param_collector.begin_forward(inputs)

        try:
            result = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            self.feat_param_collector.end_forward()
            return result
        finally:
            self.feat_param_collector.reset_forward()

    def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch: Optional[torch.Tensor | int] = None,
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if getattr(self.accelerator, "sync_gradients", True):
            self.feat_param_collector.capture_gradients(self.model)
        return loss


class FeatureParameterSFTTrainer(FeatureParameterProfilingMixin, SFTTrainer):
    pass


class FeatureParameterDPOTrainer(FeatureParameterProfilingMixin, DPOTrainer):
    pass


def save_logging_records(save_path: Optional[str], logging_records: list) -> None:
    if save_path is None:
        return

    tmp_path = f"{save_path}.tmp"
    torch.save({"logging_records": logging_records}, tmp_path)
    os.replace(tmp_path, save_path)


class FirstOrderUpdateLoggingCallback(TrainerCallback):
    finite_check_excluded_keys: tuple[str, ...] = ()

    def __init__(self,
                 collector: FeatureParameterCollector,
                 autocast_context: Callable,
                 log_every: int = 1,
                 max_steps: int = None,
                 save_path: Optional[str] = None,
                 save_every: int = 100,
                 chunk_size: int = 8):
        self.collector = collector
        self.autocast_context =autocast_context
        self.log_every = log_every
        self.max_steps = max_steps
        self.save_path = save_path
        self.save_every = save_every
        self.chunk_size = chunk_size
        self.logging_records = []

    def _save_records(self) -> None:
        save_logging_records(self.save_path, self.logging_records)

    def on_step_begin(self, args, state, control, **kwargs):
        self.collector.begin_update(state.global_step + 1, kwargs['model'])
        return control

    def _compute_first_order_update(
            self,
            inputs: dict[str, torch.Tensor | Any],
            param_before,
            model_after,
            learning_rate:float,
            assistant_only_mask:torch.Tensor,
    ):
        inputs = dict(inputs)
        hidden_before = [inputs.pop('hidden_states')]
        for layer, layer_before in zip(model_after.model.layers, param_before):
            if len(layer_before) == 0:
                hidden_before.append(layer(hidden_before[-1], **inputs))
            else:
                hidden_before.append(
                    functional_call(layer,
                                    layer_before,
                                    args=(hidden_before[-1],),
                                    kwargs=inputs,)
                )

        first_trainable = next(i for i, p in enumerate(param_before) if len(p) > 0)
        after_feature = hidden_before[first_trainable]
        for layer in model_after.model.layers[first_trainable:]:
            after_feature = layer(after_feature, **inputs)

        feature_before = model_after.model.norm(hidden_before[-1])
        feature_after = model_after.model.norm(after_feature)

        trainable = [i for i, p in enumerate(param_before) if len(p) > 0]
        feature_contribution_list = []
        for start in range(0, len(trainable), self.chunk_size):
            members = trainable[start:start + self.chunk_size]
            tangents = None
            for i in range(members[0], len(model_after.model.layers)):
                layer, layer_before = model_after.model.layers[i], param_before[i]

                # propagate existing tangent
                def propagate(hidden, _layer=layer, _before=layer_before):
                    if len(_before) == 0:
                        return _layer(hidden, **inputs)
                    return functional_call(_layer, _before, args=(hidden,), kwargs=inputs)

                if tangents is not None:
                    tangents = vmap(
                        lambda t, _h=hidden_before[i]: jvp(propagate, (_h,), (t,))[1]
                    )(tangents)

                if i in members:
                    delta_parameters_per_lr = {
                        name: (
                                (parameter - layer_before[name]) / learning_rate
                                if parameter.requires_grad
                                else torch.zeros_like(parameter)
                        )
                        for name, parameter in layer.named_parameters()
                    }

                    def perturb(parameters, _layer=layer, _h=hidden_before[i]):
                        return functional_call(_layer, parameters, args=(_h,), kwargs=inputs)

                    _, contribution = jvp(
                        perturb,
                        primals=(layer_before,),
                        tangents=(delta_parameters_per_lr,),
                    )
                    contribution = contribution.unsqueeze(0)
                    tangents = contribution if tangents is None else torch.cat(
                        [tangents, contribution], dim=0
                    )

            tangents = vmap(
                lambda t: jvp(model_after.model.norm, (hidden_before[-1],), (t,))[1]
            )(tangents)
            feature_contribution_list.append(tangents[:, assistant_only_mask])

        feature_contribution = torch.cat(feature_contribution_list, dim=0)
        return (feature_contribution.transpose(0, 1).contiguous(),
                feature_before[assistant_only_mask],
                feature_after[assistant_only_mask])

    @staticmethod
    def _trainable_layer_indices(model) -> list[int]:
        return [
            i
            for i, layer in enumerate(model.model.layers)
            if _is_trainable(layer)
        ]

    def _feature_responses(
            self,
            model,
            learning_rate: float,
    ) -> Iterator[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]]:
        for record in self.collector.records:
            with (_gradient_checkpointing_disabled(model),
                  self.autocast_context(), sdpa_kernel(SDPBackend.MATH)):
                g_tensor, feat_before, feat_after = self._compute_first_order_update(
                    inputs=record["first_layer_inputs"],
                    param_before=self.collector.model_params,
                    model_after=model,
                    learning_rate=learning_rate,
                    assistant_only_mask=record["assistant_only_mask"],
                )

            yield record, g_tensor.float(), feat_before.float(), feat_after.float()

    def _profile_step(
            self,
            model,
            learning_rate: float,
            responses: Iterable[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[dict[str, int | float], dict[str, torch.Tensor]]:
        trainable_layer_indices = self._trainable_layer_indices(model)

        per_layer_energy_sum = None
        Q_sum, C_sum, T_sum, T_rms_sum, count = 0, 0, 0, 0, 0
        feat_update_rms_sum, remainder_rms_sum = 0, 0
        gram_sum, weight_sum = 0, 0
        for record, g_tensor, feat_before, feat_after in responses:
            feat_update = feat_after - feat_before     # B, D

            T = g_tensor.sum(dim=1).square().sum().item()
            Q = g_tensor.square().sum().item()
            current_per_layer_energy_sum = g_tensor.square().sum(dim=(0,2))

            if per_layer_energy_sum is None:
                per_layer_energy_sum = current_per_layer_energy_sum
            else:
                per_layer_energy_sum += current_per_layer_energy_sum

            count += g_tensor.size(0)
            T_rms_sum += g_tensor.sum(dim=1).square().mean(dim=-1).sqrt().sum().item()
            feat_update_rms_sum += feat_update.square().mean(dim=-1).sqrt().sum().item()
            remainder_rms_sum += (feat_update - learning_rate * g_tensor.sum(dim=1)).square().mean(dim=-1).sqrt().sum().item()
            T_sum += T
            Q_sum += Q
            C_sum += T - Q

            layer_norms = g_tensor.norm(dim=-1)
            gram_sum = gram_sum + torch.bmm(g_tensor, g_tensor.transpose(1, 2)).sum(0)
            weight_sum = weight_sum + layer_norms.T @ layer_norms

        logging_dict: dict[str, int | float] = {
            'First-order feature update (L2 norm squared)': float(T_sum / count),
            'First-order feature update (RMS norm)': float(T_rms_sum / count),
            'Per-step feature update': float(feat_update_rms_sum / count),
            'Higher-order remainder': float(remainder_rms_sum / count),
            'Individual energy': float(Q_sum / count),
            'Cross energy': float(C_sum / count),
            'Gamma': float(C_sum / Q_sum),
            'Rho': float((gram_sum.sum() - gram_sum.trace())
                         / (weight_sum.sum() - weight_sum.trace())) if len(gram_sum) > 1 else 0.0,
        }

        logging_dict.update(
            {
                f'Individual energy/model.layers.{layer_idx}': float(energy_sum.item() / count)
                for layer_idx, energy_sum in zip(trainable_layer_indices, per_layer_energy_sum)
            }
        )

        tensor_records = {
            'pair_gram': gram_sum.cpu(),
            'pair_weight': weight_sum.cpu(),
        }
        return logging_dict, tensor_records

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        tmp_learning_rate = kwargs['lr_scheduler'].get_last_lr()[0]
        if tmp_learning_rate <= 0.0:
            return control

        logging_dict, tensor_records = self._profile_step(
            model=kwargs['model'],
            learning_rate=tmp_learning_rate,
            responses=self._feature_responses(kwargs['model'], tmp_learning_rate),
        )
        logging_dict = {'step': state.global_step + 1, **logging_dict}

        if not all(math.isfinite(value) for key, value in logging_dict.items()
                   if key != 'step' and key not in self.finite_check_excluded_keys):
            raise FloatingPointError("Non-finite first-order update profiling metric.")

        self.logging_records.append({
            **logging_dict,
            **tensor_records,
        })

        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.save_every > 0 and state.global_step % self.save_every == 0:
            self._save_records()

        if self.max_steps is not None and state.global_step >= self.max_steps:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.collector.remove_all()
        return control


class LossSpaceUpdateLoggingCallback(FirstOrderUpdateLoggingCallback):
    finite_check_excluded_keys = ('Quadratic-model step', 'Quadratic-model step (log2 of ratio to LR)')

    def __init__(self,
                 collector: LossSpaceCollector,
                 output_embeddings: nn.Module,
                 autocast_context: Callable,
                 token_chunk_size: int = LOSS_SPACE_TOKEN_CHUNK_SIZE,
                 **kwargs):
        super().__init__(collector=collector, autocast_context=autocast_context, **kwargs)
        if token_chunk_size <= 0:
            raise ValueError("`token_chunk_size` must be positive.")
        if collector.profile_batch_size is not None or not collector.store_gradients:
            raise ValueError("Loss-space profiling requires the whole batch to be profiled with stored gradients.")
        if any(p.requires_grad for p in output_embeddings.parameters()):
            raise ValueError("Loss-space profiling requires frozen output embeddings.")
        if getattr(output_embeddings, 'bias', None) is not None:
            raise ValueError("Loss-space profiling requires output embeddings without bias.")

        self.output_embeddings = output_embeddings
        self.token_chunk_size = token_chunk_size

    def _output_weight(self) -> torch.Tensor:
        output_weight = self.output_embeddings.weight.detach().float()
        if output_weight.ndim != 2:
            raise ValueError("Output weights must have shape (vocab_size, hidden_size).")
        return output_weight

    @staticmethod
    def _softmax_statistics(
            g_chunk: torch.Tensor,
            feat_before: torch.Tensor,
            feat_after: torch.Tensor,
            targets: torch.Tensor,
            output_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_before = nn.functional.linear(feat_before, output_weight)
        logits_after = nn.functional.linear(feat_after, output_weight)
        probs = torch.softmax(logits_before, dim=-1)
        rows = torch.arange(targets.numel(), device=targets.device)

        logp_before = logits_before[rows, targets] - torch.logsumexp(logits_before, dim=-1)
        logp_after = logits_after[rows, targets] - torch.logsumexp(logits_after, dim=-1)

        delta_logits = torch.einsum('cld,vd->clv', g_chunk, output_weight)       # c, L, V
        expected_delta = torch.einsum('cv,clv->cl', probs, delta_logits)         # E_p[dz_l]
        logp_response = delta_logits[rows, :, targets] - expected_delta
        gram = (torch.bmm(probs.unsqueeze(1) * delta_logits, delta_logits.transpose(1, 2))
                - expected_delta.unsqueeze(2) * expected_delta.unsqueeze(1))
        del delta_logits

        return logp_before, logp_after, logp_response, gram

    def _gradient_loss_response(self, model, learning_rate: float) -> torch.Tensor:
        if self.collector.gradients is None:
            raise RuntimeError("The training gradients were not captured.")

        response = []
        for layer, layer_before, layer_gradient in zip(
                model.model.layers, self.collector.model_params, self.collector.gradients):
            if len(layer_before) == 0:
                continue

            parameters_after = dict(layer.named_parameters())
            inner_product = torch.zeros((), device=next(layer.parameters()).device, dtype=torch.float64)
            for name, parameter_before in layer_before.items():
                parameter = parameters_after[name]
                gradient = layer_gradient.get(name)
                if gradient is None or not parameter.requires_grad:
                    continue
                descent_direction = (parameter_before - parameter.detach()) / learning_rate
                inner_product += (gradient.double() * descent_direction.double()).sum()
            response.append(inner_product)

        return torch.stack(response)

    def _accumulate_loss_space(
            self,
            record: dict[str, Any],
            g_tensor: torch.Tensor,
            feat_before: torch.Tensor,
            feat_after: torch.Tensor,
            output_weight: torch.Tensor,
            sums: dict[str, Any],
    ) -> None:
        targets = record['targets'].to(device=g_tensor.device)
        num_tokens, num_blocks, _ = g_tensor.shape
        if targets.numel() != num_tokens:
            raise RuntimeError("The targets must contain one token per profiled position.")

        if not sums:
            sums.update(
                count=0,
                loss_before=torch.zeros((), device=g_tensor.device, dtype=torch.float64),
                loss_after=torch.zeros((), device=g_tensor.device, dtype=torch.float64),
                response=torch.zeros(num_blocks, device=g_tensor.device, dtype=torch.float64),
                gram=torch.zeros(num_blocks, num_blocks, device=g_tensor.device, dtype=torch.float64),
            )

        for start in range(0, num_tokens, self.token_chunk_size):
            stop = start + self.token_chunk_size
            logp_before, logp_after, logp_response, gram = self._softmax_statistics(
                g_tensor[start:stop], feat_before[start:stop], feat_after[start:stop],
                targets[start:stop], output_weight,
            )
            sums['count'] += logp_before.numel()
            sums['loss_before'].sub_(logp_before.sum())
            sums['loss_after'].sub_(logp_after.sum())
            sums['response'].add_(logp_response.sum(dim=0))
            sums['gram'].add_(gram.sum(dim=0))

    def _loss_space_responses(
            self,
            responses: Iterable[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]],
            output_weight: torch.Tensor,
            sums: dict[str, Any],
    ) -> Iterator[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Accumulates the loss-space sums of every micro-batch and passes the responses on to the parent."""
        for record, g_tensor, feat_before, feat_after in responses:
            self._accumulate_loss_space(record, g_tensor, feat_before, feat_after, output_weight, sums)
            yield record, g_tensor, feat_before, feat_after

    def _loss_space_logging_dict(
            self,
            sums: dict[str, Any],
            response_gradient: torch.Tensor,
            learning_rate: float,
            trainable_layer_indices: list[int],
    ) -> tuple[dict[str, int | float], dict[str, torch.Tensor]]:
        count = sums['count']
        response = sums['response'] / count      # d_l: first-order loss decrease per unit LR of block l
        gram = sums['gram'] / count              # loss-metric Gram of the blocks
        loss_before = float(sums['loss_before'] / count)
        loss_after = float(sums['loss_after'] / count)

        D1 = float(response.sum())
        C2 = float(gram.sum())
        Q = float(gram.trace())
        eta_quad = D1 / C2 if C2 > 0.0 else float('nan')

        logging_dict: dict[str, int | float] = {
            'First-order loss decrease': D1,
            'First-order loss decrease (gradient)': float(response_gradient.sum()),
            'Loss-metric individual energy': Q,
            'Loss-metric cross energy': C2 - Q,
            'Loss-metric curvature': C2,
            'Gamma (loss metric)': (C2 - Q) / Q,
            'Quadratic-model step': eta_quad,
            'Quadratic-model step (log2 of ratio to LR)': (
                math.log2(eta_quad / learning_rate) if eta_quad > 0.0 else float('nan')
            ),
            'Loss before': loss_before,
            'Loss after': loss_after,
            'Realized loss change': loss_after - loss_before,
            'First-order loss change': -learning_rate * D1,
            'Quadratic loss change': -learning_rate * D1 + 0.5 * learning_rate ** 2 * C2,
        }
        logging_dict.update(
            {
                f'First-order loss decrease/model.layers.{layer_idx}': float(value)
                for layer_idx, value in zip(trainable_layer_indices, response.tolist())
            }
        )
        logging_dict.update(
            {
                f'First-order loss decrease (gradient)/model.layers.{layer_idx}': float(value)
                for layer_idx, value in zip(trainable_layer_indices, response_gradient.tolist())
            }
        )

        tensor_records = {
            'loss_response_per_block': response.float().cpu(),
            'loss_response_per_block_gradient': response_gradient.float().cpu(),
            'loss_gram': gram.float().cpu(),
        }
        return logging_dict, tensor_records

    def _profile_step(self, model, learning_rate, responses):
        output_weight = self._output_weight()
        sums: dict[str, Any] = {}

        logging_dict, tensor_records = super()._profile_step(
            model=model,
            learning_rate=learning_rate,
            responses=self._loss_space_responses(responses, output_weight, sums),
        )
        if not sums:
            raise RuntimeError("No valid tokens were captured for loss-space profiling.")

        loss_logging_dict, loss_tensor_records = self._loss_space_logging_dict(
            sums=sums,
            response_gradient=self._gradient_loss_response(model, learning_rate),
            learning_rate=learning_rate,
            trainable_layer_indices=self._trainable_layer_indices(model),
        )
        logging_dict.update(loss_logging_dict)
        tensor_records.update(loss_tensor_records)

        return logging_dict, tensor_records


class LossSpaceDPOLoggingCallback(LossSpaceUpdateLoggingCallback):
    def __init__(self, *args, beta: float, **kwargs):
        super().__init__(*args, **kwargs)
        if beta <= 0.0:
            raise ValueError("`beta` must be positive.")
        self.beta = beta

    def _accumulate_loss_space(
            self,
            record: dict[str, Any],
            g_tensor: torch.Tensor,
            feat_before: torch.Tensor,
            feat_after: torch.Tensor,
            output_weight: torch.Tensor,
            sums: dict[str, Any],
    ) -> None:
        device = g_tensor.device
        mask = record['assistant_only_mask'].to(device=device)
        targets = record['targets'].to(device=device)
        num_tokens, num_blocks, _ = g_tensor.shape
        if targets.numel() != num_tokens:
            raise RuntimeError("The targets must contain one token per profiled position.")
        if mask.shape[0] % 2 != 0:
            raise RuntimeError("Expected a concatenated [chosen; rejected] batch with an even number of sequences.")

        num_pairs = mask.shape[0] // 2
        sequence_of_token = mask.nonzero(as_tuple=False)[:, 0]
        chosen = sequence_of_token < num_pairs
        pair_of_token = torch.where(chosen, sequence_of_token, sequence_of_token - num_pairs)
        sign_of_token = torch.where(chosen, 1.0, -1.0).to(device=device, dtype=torch.float32)

        ref_margin = record['ref_chosen_logps'].to(device) - record['ref_rejected_logps'].to(device)
        if ref_margin.numel() != num_pairs:
            raise RuntimeError("The reference log-probs must contain one entry per profiled pair.")

        logp_before_pair = torch.zeros(num_pairs, device=device)
        logp_after_pair = torch.zeros(num_pairs, device=device)
        response_pair = torch.zeros(num_pairs, num_blocks, device=device)
        gram_pair = torch.zeros(num_pairs, num_blocks, num_blocks, device=device)
        for start in range(0, num_tokens, self.token_chunk_size):
            stop = start + self.token_chunk_size
            logp_before, logp_after, logp_response, gram = self._softmax_statistics(
                g_tensor[start:stop], feat_before[start:stop], feat_after[start:stop],
                targets[start:stop], output_weight,
            )
            pair, sign = pair_of_token[start:stop], sign_of_token[start:stop]
            logp_before_pair.index_add_(0, pair, sign * logp_before)
            logp_after_pair.index_add_(0, pair, sign * logp_after)
            response_pair.index_add_(0, pair, sign[:, None] * logp_response)
            gram_pair.index_add_(0, pair, sign[:, None, None] * gram)

        margin_before = self.beta * (logp_before_pair - ref_margin)
        margin_after = self.beta * (logp_after_pair - ref_margin)
        sigma = torch.sigmoid(margin_before)
        sigma_neg = 1.0 - sigma

        u = self.beta * response_pair
        response = sigma_neg[:, None] * u
        gram_margin = (sigma * sigma_neg)[:, None, None] * u[:, :, None] * u[:, None, :]
        gram_token = (sigma_neg * self.beta)[:, None, None] * gram_pair
        loss_before = -nn.functional.logsigmoid(margin_before)
        loss_after = -nn.functional.logsigmoid(margin_after)

        if not sums:
            sums.update(
                count=0,
                loss_before=torch.zeros((), device=device, dtype=torch.float64),
                loss_after=torch.zeros((), device=device, dtype=torch.float64),
                response=torch.zeros(num_blocks, device=device, dtype=torch.float64),
                gram=torch.zeros(num_blocks, num_blocks, device=device, dtype=torch.float64),
                gram_margin=torch.zeros((), device=device, dtype=torch.float64),
                gram_token=torch.zeros((), device=device, dtype=torch.float64),
                margin=torch.zeros((), device=device, dtype=torch.float64),
                active=torch.zeros((), device=device, dtype=torch.float64),
            )

        sums['count'] += num_pairs
        sums['loss_before'].add_(loss_before.sum())
        sums['loss_after'].add_(loss_after.sum())
        sums['response'].add_(response.sum(dim=0))
        sums['gram'].add_((gram_margin + gram_token).sum(dim=0))
        sums['gram_margin'].add_(gram_margin.sum())
        sums['gram_token'].add_(gram_token.sum())
        sums['margin'].add_(margin_before.sum())
        sums['active'].add_(sigma_neg.sum())

    def _loss_space_logging_dict(self, sums, response_gradient, learning_rate, trainable_layer_indices):
        logging_dict, tensor_records = super()._loss_space_logging_dict(
            sums=sums,
            response_gradient=response_gradient,
            learning_rate=learning_rate,
            trainable_layer_indices=trainable_layer_indices,
        )
        count = sums['count']
        logging_dict.update(
            {
                'Loss-metric curvature (margin part)': float(sums['gram_margin'] / count),
                'Loss-metric curvature (token part)': float(sums['gram_token'] / count),
                'Margin': float(sums['margin'] / count),
                'Active fraction': float(sums['active'] / count),
            }
        )
        return logging_dict, tensor_records


def adamw_update_direction(optimizer, parameter: nn.Parameter, parameter_before: torch.Tensor) -> torch.Tensor:
    group = next(g for g in optimizer.param_groups if any(p is parameter for p in g['params']))
    state = optimizer.state[parameter]
    step = float(state['step'])
    beta1, beta2 = group['betas']
    momentum = state['exp_avg'].float() / (1.0 - beta1 ** step)
    denominator = state['exp_avg_sq'].float().sqrt() / math.sqrt(1.0 - beta2 ** step) + group['eps']
    return momentum / denominator + group['weight_decay'] * parameter_before.float()


class RoundingScanLoggingCallback(TrainerCallback):
    def __init__(self,
                 collector: FeatureParameterCollector,
                 measure_every: int = 500,
                 scan_exponents: Iterable[float] = range(8, 27),
                 max_steps: int = None,
                 save_path: Optional[str] = None):
        if measure_every <= 0:
            raise ValueError("`measure_every` must be positive.")
        if collector.profile_batch_size is not None:
            raise ValueError("The rounding scan uses the training gradient, so the whole batch must be profiled.")

        self.collector = collector
        self.measure_every = measure_every
        self.scan_exponents = torch.tensor(list(scan_exponents), dtype=torch.float64)
        self.max_steps = max_steps
        self.save_path = save_path
        self.logging_records = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.collector.begin_update(state.global_step + 1, kwargs['model'])
        self.collector.store_gradients = state.global_step % self.measure_every == 0
        return control

    @torch.no_grad()
    def _scan_block(self, layer, layer_before, layer_gradient, optimizer):
        etas = 2.0 ** -self.scan_exponents
        device = next(layer.parameters()).device
        nominal = torch.zeros((), device=device, dtype=torch.float64)
        applied = torch.zeros((), device=device, dtype=torch.float64)
        rounded = torch.zeros(len(etas), device=device, dtype=torch.float64)
        moved = torch.zeros(len(etas), device=device, dtype=torch.float64)
        numel = torch.zeros((), device=device, dtype=torch.float64)
        for name, parameter in layer.named_parameters():
            gradient = layer_gradient.get(name)
            if gradient is None:
                continue

            gradient = gradient.float()
            theta = layer_before[name].float()
            direction = adamw_update_direction(optimizer, parameter, theta)
            nominal += (gradient * direction).sum(dtype=torch.float64)
            applied += (gradient * (theta - parameter.float())).sum(dtype=torch.float64)

            theta_bf16 = theta.to(torch.bfloat16).float()
            for k, eta in enumerate(etas.tolist()):
                delta = (theta_bf16 - eta * direction).to(torch.bfloat16).float() - theta_bf16
                rounded[k] -= (gradient * delta).sum(dtype=torch.float64) / eta
                moved[k] += delta.ne(0.0).sum()
            numel += parameter.numel()

        return nominal, applied, rounded, moved, numel

    @torch.no_grad()
    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.collector.store_gradients:
            return control
        if self.collector.gradients is None:
            raise RuntimeError("The training gradients were not captured.")

        model, optimizer = kwargs['model'], kwargs['optimizer']
        learning_rate = kwargs['lr_scheduler'].get_last_lr()[0]
        layer_indices, scans = [], []
        for layer_idx, (layer, layer_before, layer_gradient) in enumerate(
                zip(model.model.layers, self.collector.model_params, self.collector.gradients)):
            if len(layer_before) == 0:
                continue
            layer_indices.append(layer_idx)
            scans.append(self._scan_block(layer, layer_before, layer_gradient, optimizer))

        nominal, applied, rounded, moved, numel = (torch.stack(values) for values in zip(*scans))
        realized = applied / learning_rate if learning_rate > 0.0 else torch.full_like(applied, float('nan'))
        exponents = self.scan_exponents.tolist()

        logging_dict: dict[str, int | float] = {
            'step': state.global_step,
            'lr': float(learning_rate),
            'First-order loss decrease': float(nominal.sum()),
            'First-order loss decrease (realized)': float(realized.sum()),
        }
        logging_dict.update(
            {
                f'First-order loss decrease/model.layers.{layer_idx}': value
                for layer_idx, value in zip(layer_indices, nominal.tolist())
            }
        )
        logging_dict.update(
            {
                f'Rounded ratio/2^-{exponent:g}': value
                for exponent, value in zip(exponents, (rounded.sum(dim=0) / nominal.sum()).tolist())
            }
        )
        logging_dict.update(
            {
                f'Mobile fraction/2^-{exponent:g}': value
                for exponent, value in zip(exponents, (moved.sum(dim=0) / numel.sum()).tolist())
            }
        )
        self.logging_records.append({
            **logging_dict,
            'scan_exponents': self.scan_exponents.clone(),
            'loss_response_per_block': nominal.cpu(),
            'loss_response_per_block_realized': realized.cpu(),
            'loss_response_per_block_rounded': rounded.cpu(),
            'ratio_per_block': (rounded / nominal[:, None]).cpu(),
            'mobile_fraction_per_block': (moved / numel[:, None]).cpu(),
        })

        save_logging_records(self.save_path, self.logging_records)

        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_steps is not None and state.global_step >= self.max_steps:
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.collector.remove_all()
        return control
