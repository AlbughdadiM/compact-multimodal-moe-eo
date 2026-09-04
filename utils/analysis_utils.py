"""Small, public-API-based helpers used by the active analysis notebooks."""

from contextlib import contextmanager, nullcontext

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt


@contextmanager
def deterministic_routing(model):
    """Temporarily disable router noise without changing other module modes."""
    gates = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "NoisyTopKGate"
    ]
    training_states = [gate.training for gate in gates]
    for gate in gates:
        gate.eval()
    try:
        yield
    finally:
        for gate, was_training in zip(gates, training_states):
            gate.train(was_training)


@contextmanager
def _evaluation_mode(model):
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in training_states:
            module.training = was_training


def _fine_routing(features, layer_index, prefusion_modality=None):
    routing_layers = features["routing"]
    if layer_index < 0:
        layer_index += len(routing_layers)
    if not 0 <= layer_index < len(routing_layers):
        raise IndexError(f"layer_index {layer_index} is outside the encoder")

    routing = routing_layers[layer_index]
    if routing["stage"] == "pre_fusion":
        available = list(routing["by_modality"])
        modality = prefusion_modality or available[0]
        if modality not in routing["by_modality"]:
            raise ValueError(
                f"Pre-fusion modality '{modality}' is unavailable; choose one of {available}"
            )
        return routing["by_modality"][modality], modality, layer_index

    fine_start = int(features["token_layout"]["num_meta_tokens"]) + 1
    fine_count = int(features["token_layout"]["num_fine_tokens"])
    fine_end = fine_start + fine_count
    fine_routing = {
        key: value[:, fine_start:fine_end]
        if torch.is_tensor(value) and value.ndim >= 3
        else value
        for key, value in routing.items()
        if key != "stage"
    }
    return fine_routing, "fused", layer_index


def _expert_colors(num_experts):
    base = mpl.colormaps.get_cmap("tab20")(np.arange(20))
    return np.vstack([base] * int(np.ceil(num_experts / 20)))[:num_experts]


def _plot_assignment(base_rgb, assignments, patch_size, num_experts, title):
    patch_height, patch_width = assignments.shape
    expected_shape = (patch_height * patch_size, patch_width * patch_size)
    if tuple(base_rgb.shape[:2]) != expected_shape:
        raise ValueError(
            f"RGB shape {base_rgb.shape[:2]} does not match patch layout {expected_shape}"
        )

    colors = _expert_colors(num_experts)
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    axis.imshow(base_rgb)
    y, x = np.mgrid[0:patch_height, 0:patch_width]
    axis.scatter(
        (x.reshape(-1) + 0.5) * patch_size,
        (y.reshape(-1) + 0.5) * patch_size,
        c=colors[assignments.reshape(-1)],
        s=18,
        edgecolors="black",
        linewidths=0.25,
    )
    axis.set_title(title)
    axis.axis("off")
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[index],
            markeredgecolor="black",
            markersize=6,
            label=f"E{index}",
        )
        for index in range(num_experts)
    ]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    figure.tight_layout()
    plt.show()


def _plot_gate_maps(base_rgb, gate_maps, patch_size, route_counts, title):
    num_experts = gate_maps.shape[0]
    figure, axes = plt.subplots(num_experts, 2, figsize=(10, 3 * num_experts))
    if num_experts == 1:
        axes = np.expand_dims(axes, axis=0)
    maximum = max(float(gate_maps.max()), 1e-6)
    image = None
    for expert_index, patch_map in enumerate(gate_maps):
        pixel_map = np.repeat(
            np.repeat(patch_map, patch_size, axis=0), patch_size, axis=1
        )
        axes[expert_index, 0].imshow(base_rgb)
        image = axes[expert_index, 0].imshow(
            pixel_map, cmap="magma", vmin=0.0, vmax=maximum, alpha=0.45
        )
        axes[expert_index, 0].set_title(
            f"E{expert_index} overlay ({int(route_counts[expert_index])} active routes)"
        )
        axes[expert_index, 1].imshow(
            patch_map, cmap="magma", vmin=0.0, vmax=maximum
        )
        axes[expert_index, 1].set_title(f"E{expert_index} gate weight")
        for axis in axes[expert_index]:
            axis.axis("off")
    figure.suptitle(title)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Executed gate weight")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


@torch.no_grad()
def layer_report_multimodal(
    model,
    raster_dict,
    raster_band_names,
    base_rgb,
    meta_dict=None,
    raster_valid_masks=None,
    meta_valid_masks=None,
    image_index=0,
    layer_index=-1,
    token_group="fine",
    deterministic=True,
    prefusion_modality=None,
    title_prefix="",
):
    """Plot true top-k routes and gate weights for one encoder layer."""
    if token_group != "fine":
        raise ValueError("Only fine patch tokens exist in the simplified encoder")

    context = deterministic_routing(model) if deterministic else nullcontext()
    with _evaluation_mode(model), context:
        features = model.forward_features(
            raster_dict=raster_dict,
            raster_valid_masks=raster_valid_masks,
            raster_band_names=raster_band_names,
            meta_dict=meta_dict,
            meta_valid_masks=meta_valid_masks,
            return_routing=True,
            stochastic_routing=not deterministic,
        )

    routing, stream_name, resolved_layer = _fine_routing(
        features, layer_index, prefusion_modality=prefusion_modality
    )
    gates = routing["gates"].float().cpu().numpy()
    batch_size, num_patches, num_experts = gates.shape
    if not 0 <= image_index < batch_size:
        raise IndexError(f"image_index {image_index} is outside the batch")

    fine_height = int(features["token_layout"]["fine_height"])
    fine_width = int(features["token_layout"]["fine_width"])
    if num_patches != fine_height * fine_width:
        raise ValueError("Routing patch count does not match the runtime grid")

    assignments = gates.argmax(axis=-1).reshape(batch_size, fine_height, fine_width)
    gate_maps = gates.transpose(0, 2, 1).reshape(
        batch_size, num_experts, fine_height, fine_width
    )
    route_counts = (gates > 0).sum(axis=1)
    label = f"Layer {resolved_layer} ({stream_name})"
    _plot_assignment(
        base_rgb,
        assignments[image_index],
        model.encoder.patch_size,
        num_experts,
        f"{title_prefix}{label} - top-1 expert",
    )
    _plot_gate_maps(
        base_rgb,
        gate_maps[image_index],
        model.encoder.patch_size,
        route_counts[image_index],
        f"{title_prefix}{label} - executed routing weights",
    )
    return {
        "num_experts": num_experts,
        "usage_batch": route_counts,
        "usage_image": route_counts[image_index],
        "assignment_maps": assignments,
        "gate_maps": gate_maps,
        "layer_index": resolved_layer,
        "stream": stream_name,
        "moe_loss": float(features["moe_loss"]),
    }


def month_from_dates(dates):
    """Convert YYYY-MM-DD-like values to month integers, or -1 if malformed."""
    months = []
    for value in dates:
        text = "" if value is None else str(value)
        try:
            months.append(int(text[5:7]))
        except (TypeError, ValueError):
            months.append(-1)
    return np.asarray(months, dtype=np.int64)


def majority_class_per_sample(label_tensor, ignore_values=None):
    """Return the most frequent valid class in each label raster."""
    labels = np.asarray(label_tensor)
    if labels.ndim == 4:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError("label_tensor must have shape [B, 1, H, W] or [B, H, W]")

    ignored = set(ignore_values or [])
    majorities = []
    for sample in labels:
        valid = [value for value in sample.reshape(-1) if value not in ignored]
        if not valid:
            majorities.append(-1)
            continue
        values, counts = np.unique(np.asarray(valid, dtype=np.int64), return_counts=True)
        majorities.append(int(values[counts.argmax()]))
    return np.asarray(majorities, dtype=np.int64)
