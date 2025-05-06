import torch
from torchvision.models import mobilenet_v2
from torchvision.models.quantization import mobilenet_v2 as mobilenet_v2_q
import argparse

def export_model(model: torch.nn.Module, dummy: torch.Tensor, onnx_path: str):
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"[✓] Exported ONNX: {onnx_path}")


def main():
    parser = argparse.ArgumentParser("Export TorchVision MobileNetV2 to ONNX (FP32 & INT8)")
    parser.add_argument(
        "--quantized", action="store_true",
        help="Export the quantized INT8 model instead of FP32"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the ONNX file"
    )
    args = parser.parse_args()

    # Create a dummy input matching MobileNetV2's expected shape
    dummy = torch.randn(1, 3, 224, 224)

    if args.quantized:
        # Built-in quantized MobileNetV2 from torchvision
        model = mobilenet_v2_q(pretrained=True, quantize=True).eval()
    else:
        # Standard FP32 MobileNetV2
        model = mobilenet_v2(pretrained=True).eval()

    export_model(model, dummy, args.output)


if __name__ == "__main__":
    main()
