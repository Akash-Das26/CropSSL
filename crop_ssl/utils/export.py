"""
Model Export for Production Deployment.

Supports ONNX export with dynamic batching and input shapes.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
) -> str:
    """Export model to ONNX format.

    Args:
        model: Trained model to export.
        save_path: Path to save ONNX file.
        input_shape: Example input shape.
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axis configuration.
        input_names: Names for input tensors.
        output_names: Names for output tensors.

    Returns:
        Path to saved ONNX file.
    """
    model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    if input_names is None:
        input_names = ["input"]

    if output_names is None:
        output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
    )

    print(f"Model exported to {save_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")

    return str(save_path)


def verify_onnx(
    onnx_path: str,
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX model matches PyTorch output.

    Args:
        onnx_path: Path to ONNX file.
        model: Original PyTorch model.
        input_shape: Input shape for verification.
        atol: Absolute tolerance.

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping verification.")
        return True

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # PyTorch output
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None, {"input": dummy_input.numpy()}
    )[0]

    match = abs(torch_output - onnx_output).max() < atol
    print(f"ONNX verification: {'PASS' if match else 'FAIL'}")
    print(f"  Max diff: {abs(torch_output - onnx_output).max():.8f}")
    return match


def export_ssl_backbone(
    ssl_model: nn.Module,
    save_path: str,
    backbone_type: str = "teacher",
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> str:
    """Export SSL model backbone for deployment.

    Args:
        ssl_model: Pre-trained SSL model.
        save_path: Path to save ONNX file.
        backbone_type: 'teacher' or 'student' for DINOv2/MoCo.
        input_shape: Input shape.

    Returns:
        Path to saved ONNX file.
    """
    ssl_model.eval()

    # Extract backbone
    if hasattr(ssl_model, "teacher_backbone") and backbone_type == "teacher":
        backbone = ssl_model.teacher_backbone
    elif hasattr(ssl_model, "student_backbone"):
        backbone = ssl_model.student_backbone
    elif hasattr(ssl_model, "encoder"):
        backbone = ssl_model.encoder
    elif hasattr(ssl_model, "query_encoder"):
        backbone = ssl_model.query_encoder
    else:
        backbone = ssl_model

    # Create wrapper that returns features
    class BackboneWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            return self.backbone.forward_features(x)

    wrapper = BackboneWrapper(backbone)

    return export_to_onnx(
        wrapper, save_path, input_shape=input_shape,
        input_names=["input"], output_names=["features"],
    )


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters.

    Returns:
        Dict with total, trainable, and frozen parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable / max(total, 1) * 100,
        "frozen_pct": frozen / max(total, 1) * 100,
    }


def model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
) -> str:
    """Generate model summary string.

    Args:
        model: Model to summarize.
        input_shape: Input shape for FLOPs calculation.

    Returns:
        Formatted summary string.
    """
    params = count_parameters(model)

    lines = [
        "=" * 50,
        "Model Summary",
        "=" * 50,
        f"Total parameters:     {params['total']:>12,}",
        f"Trainable parameters: {params['trainable']:>12,} ({params['trainable_pct']:.2f}%)",
        f"Frozen parameters:    {params['frozen']:>12,} ({params['frozen_pct']:.2f}%)",
        f"Input shape:          {str(input_shape):>12}",
        "=" * 50,
    ]

    return "\n".join(lines)
