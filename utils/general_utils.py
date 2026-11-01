from transformers import AutoTokenizer, PreTrainedModel


def configure_trainable_layers(model: PreTrainedModel,
                               last_n: int,):
    if last_n == 0:
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)

        model_n_layer = model.config.num_hidden_layers
        if not (0 < last_n <= model_n_layer):
            raise ValueError(f"Last N must be between 1 and {model_n_layer}.")

        prefix_list = [f"model.layers.{l_idx}." for l_idx in range(model_n_layer - last_n, model_n_layer)]
        
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefix_list):
                p.requires_grad_(True)
