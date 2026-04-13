"""
Run SAM3 image inference from the command line.

Example:
  python scripts/image_infer.py \
      --checkpoint /root/SAM3-for-Med/checkpoints/sam3.safetensors \
      --image /root/SAM3-for-Med/assets/images/truck.jpg \
      --prompt truck
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


COLORS = [
    (255, 59, 48),
    (52, 199, 89),
    (10, 132, 255),
    (255, 159, 10),
    (191, 90, 242),
    (255, 55, 95),
    (90, 200, 250),
    (48, 209, 88),
]


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 image inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pt/.pth/.safetensors). Uses Hugging Face if omitted.",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt, for example "truck" or "lesion"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to overlay image. Defaults to outputs/image_infer/<image>_<prompt>.png",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for keeping masks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Inference device, for example "cuda" or "cpu"',
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save raw predictions as JSON next to the output image",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=0,
        help="Maximum number of detections to render. 0 means keep all.",
    )
    return parser.parse_args()


def make_output_path(args):
    if args.output is not None:
        return Path(args.output)
    safe_prompt = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.prompt)
    image_name = Path(args.image).stem
    return Path("outputs") / "image_infer" / f"{image_name}_{safe_prompt}.png"


def tensor_to_numpy(state, max_masks=0):
    scores = state["scores"].detach().float().cpu().numpy()
    boxes = state["boxes"].detach().float().cpu().numpy()
    masks = state["masks"].detach().cpu().numpy()

    if max_masks > 0:
        scores = scores[:max_masks]
        boxes = boxes[:max_masks]
        masks = masks[:max_masks]
    return scores, boxes, masks


def render_overlay(image, scores, boxes, masks, prompt):
    base = np.array(image.convert("RGB"), dtype=np.uint8)
    overlay = base.copy().astype(np.float32)

    for idx, mask in enumerate(masks):
        color = np.array(COLORS[idx % len(COLORS)], dtype=np.float32)
        mask_bool = mask[0].astype(bool)
        if mask_bool.any():
            overlay[mask_bool] = overlay[mask_bool] * 0.6 + color * 0.4

    output_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output_image)

    for idx, (score, box) in enumerate(zip(scores, boxes)):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        color = COLORS[idx % len(COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), f"{prompt}: {score:.3f}", fill=color)

    return output_image


def save_predictions_json(path, image_path, prompt, scores, boxes):
    payload = {
        "image": str(image_path),
        "prompt": prompt,
        "num_detections": int(len(scores)),
        "scores": [float(v) for v in scores.tolist()],
        "boxes_xyxy": [[float(x) for x in box.tolist()] for box in boxes],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    image_path = Path(args.image)
    output_path = make_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_from_hf = args.checkpoint is None
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=load_from_hf,
        device=args.device,
        eval_mode=True,
    )
    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.threshold,
    )

    image = Image.open(image_path).convert("RGB")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with autocast_ctx:
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

    scores, boxes, masks = tensor_to_numpy(state, max_masks=args.max_masks)
    output_image = render_overlay(image, scores, boxes, masks, args.prompt)
    output_image.save(output_path)

    print(f"image={image_path}")
    print(f"prompt={args.prompt}")
    print(f"num_detections={len(scores)}")
    print(f"output={output_path.resolve()}")
    for idx, (score, box) in enumerate(zip(scores, boxes)):
        print(
            f"[{idx}] score={float(score):.4f} "
            f"box={[round(float(v), 2) for v in box.tolist()]}"
        )

    if args.save_json:
        json_path = output_path.with_suffix(".json")
        save_predictions_json(
            json_path,
            image_path=image_path,
            prompt=args.prompt,
            scores=scores,
            boxes=boxes,
        )
        print(f"json={json_path.resolve()}")


if __name__ == "__main__":
    main()
