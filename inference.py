
import os
import re
import json
import time
from pathlib import Path

import numpy as np
import torch
import fire
from PIL import Image, ImageDraw, ImageFont

# #region agent log
_DBG_LOG = "./logs/debug-6a42f9.log"


def _agent_dbg(hypothesis_id, location, message, data, run_id="pre-fix"):
    try:
        with open(_DBG_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "6a42f9",
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

# ---------------------------------------------------------------------------
# Helpers – device
# ---------------------------------------------------------------------------


def move_inputs_to_device(model_inputs: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in model_inputs.items()}


def get_model_inputs(
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
) -> dict:
    image = Image.open(image_file_path).convert("RGB")
    # #region agent log
    _agent_dbg(
        "D",
        "inference.py:get_model_inputs",
        "original vs model image size",
        {
            "orig_w": image.size[0],
            "orig_h": image.size[1],
            "aspect": round(image.size[0] / max(image.size[1], 1), 4),
            "model_img_size": getattr(processor, "image_size", None),
            "image_seq_length": getattr(processor, "image_seq_length", None),
            "prompt": prompt,
        },
    )
    # #endregion
    model_inputs = processor(text=[prompt], images=[image])
    return move_inputs_to_device(model_inputs, device)


# ---------------------------------------------------------------------------
# Helpers – segmentation mask overlay
# ---------------------------------------------------------------------------


def decode_segmentation_mask(
    seg_tokens: list[int],
    vae_weights_dir: str,
    device: str,
) -> np.ndarray:
    """
    Run the OID VQ-VAE decoder on 16 codebook indices.

    Returns:
        mask_np: (64, 64) float32 array with values in [0, 1].
    """
    from variational_autoencoder import load_vae_decoder

    vae = load_vae_decoder(vae_weights_dir, device=device)

    token_tensor = torch.tensor(
        [seg_tokens], dtype=torch.long, device=device
    )  # (1, 16)
    with torch.no_grad():
        mask = vae(token_tensor)  # (1, 1, 64, 64)

    mask_np = mask.squeeze().cpu().float().numpy()  # (64, 64) in [0, 1]
    # #region agent log
    flat = mask_np.reshape(-1)
    _agent_dbg(
        "C",
        "inference.py:decode_segmentation_mask",
        "vae mask stats",
        {
            "seg_tokens": seg_tokens,
            "n_tokens": len(seg_tokens),
            "mask_shape": list(mask_np.shape),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "frac_ge_0.5": float(np.mean(flat >= 0.5)),
            "frac_ge_0.1": float(np.mean(flat >= 0.1)),
        },
    )
    # #endregion
    return mask_np


def overlay_mask_on_image(
    original_image: Image.Image,
    mask_np: np.ndarray,
    bbox_locs: list[int] | None = None,
    mask_color: tuple[int, int, int] = (0, 255, 100),
    alpha: float = 0.55,
    threshold: float = 0.5,
) -> Image.Image:
    """Overlay a mask onto an image; optionally restrict to a bounding box.

    Args:
        original_image: Source RGB image.
        mask_np: (64, 64) float32 mask in [0, 1].
        bbox_locs: Optional list of four ints ``[y_min, x_min, y_max, x_max]`` in the
            0‑1023 coordinate space that defines where the mask should be placed.
            If ``None`` the mask covers the full image.
        mask_color: Colour for the mask region.
        alpha: Opacity of the overlay.
        threshold: Threshold for mask binarisation.
    """
    """
    Resize *mask_np* to match the bounding box defined by *bbox_locs*, threshold it, and
    draw a semi-transparent coloured overlay on top of the image.

    Args:
        original_image: The source PIL image (RGB).
        mask_np: (64, 64) float32 array in [0, 1].
        bbox_locs: List of parsed <loc> tokens [y_min, x_min, y_max, x_max].
        mask_color: RGB colour for the positive-mask region.
        alpha: Opacity of the mask overlay (0 = invisible, 1 = opaque).
        threshold: Pixels above this value are treated as foreground.
    """
    img_w, img_h = original_image.size

    # Default bounding box is the whole image (fallback)
    x1, y1, x2, y2 = 0, 0, img_w, img_h

    # If locs are provided, calculate the actual bounding box
    if bbox_locs and len(bbox_locs) >= 4:
        y_min_val, x_min_val, y_max_val, x_max_val = bbox_locs[:4]
        # Convert PaliGemma 1024-bin coordinates to absolute pixels
        y1 = int((y_min_val / 1024.0) * img_h)
        x1 = int((x_min_val / 1024.0) * img_w)
        y2 = int((y_max_val / 1024.0) * img_h)
        x2 = int((x_max_val / 1024.0) * img_w)

        # Ensure valid bounds within the image
        x1, x2 = max(0, min(x1, x2)), min(img_w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(img_h, max(y1, y2))

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    # #region agent log
    _agent_dbg(
        "A",
        "inference.py:overlay_mask_on_image",
        "bbox mapping",
        {
            "bbox_locs": bbox_locs,
            "n_locs": 0 if bbox_locs is None else len(bbox_locs),
            "used_default_full_image": not (bbox_locs and len(bbox_locs) >= 4),
            "img_w": img_w,
            "img_h": img_h,
            "pixel_xyxy": [x1, y1, x2, y2],
            "box_w": box_w,
            "box_h": box_h,
            "box_frac_w": round(box_w / max(img_w, 1), 4),
            "box_frac_h": round(box_h / max(img_h, 1), 4),
            "box_area_frac": round((box_w * box_h) / max(img_w * img_h, 1), 4),
        },
    )
    # #endregion

    # 1. Resize mask to the bounding box dimensions
    mask_pil = Image.fromarray(
        (mask_np * 255).astype(np.uint8)
    )  # auto-mode "L" for 2-D uint8
    mask_pil = mask_pil.resize((box_w, box_h), resample=Image.BILINEAR)
    mask_arr = np.array(mask_pil) / 255.0  # back to [0,1]

    # 2. Build RGBA overlay for the entire canvas
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)

    # 3. Apply threshold and assign color to the bounding box region
    fg = mask_arr >= threshold
    region_arr = np.zeros((box_h, box_w, 4), dtype=np.uint8)
    region_arr[fg] = (*mask_color, int(alpha * 255))

    # 4. Paste the local mask region into the full image overlay
    overlay_arr[y1 : y1 + box_h, x1 : x1 + box_w] = region_arr
    overlay = Image.fromarray(overlay_arr)

    # 5. Composite onto a copy of the original
    base = original_image.convert("RGBA")
    composited = Image.alpha_composite(base, overlay)
    return composited.convert("RGB")


def save_overlay(image_file_path, overlaid, suffix="_seg_mask"):
    """Save an overlaid image alongside the original.

    Args:
        image_file_path: Path to the original image.
        overlaid: PIL image with overlay applied.
        suffix: Suffix to append before the file extension (e.g. "_seg_mask" or "_det").
    """
    filename = os.path.basename(image_file_path)
    name_only, ext = os.path.splitext(filename)
    new_filename = f"{name_only}{suffix}{ext}"
    out_path = os.path.join("/kaggle/working", new_filename)
    overlaid.save(out_path)
    return out_path
def draw_bboxes_on_image(original_image: Image.Image, boxes: list[list[int]], box_color: tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Image.Image:
    """Draw bounding boxes on a copy of the image and return it.

    Args:
        original_image: PIL Image in RGB mode.
        boxes: List of bounding box token groups, each a list of four ints
            ``[y_min, x_min, y_max, x_max]`` in the 0‑1023 coordinate space.
        box_color: RGB colour for the rectangle outline.
        thickness: Line thickness in pixels.
    """
    img = original_image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    for box in boxes:
        if len(box) < 4:
            continue
        y_min, x_min, y_max, x_max = box[:4]
        x1 = int((x_min / 1024.0) * img_w)
        y1 = int((y_min / 1024.0) * img_h)
        x2 = int((x_max / 1024.0) * img_w)
        y2 = int((y_max / 1024.0) * img_h)
        # Draw multiple rectangles for thickness
        for i in range(thickness):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=box_color)
    return img

# ---------------------------------------------------------------------------
# Helpers – sampling
# ---------------------------------------------------------------------------


def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    vae_weights_dir: str | None = None,
) -> None:

    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    kv_cache = KVCache()

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens: list[torch.Tensor] = []

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # → (1,)
        generated_tokens.append(next_token)

        if next_token.item() == stop_token:
            break

        input_ids = next_token.unsqueeze(-1)  # (1, 1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
            ],
            dim=-1,
        )
        pixel_values = None  # vision tower only needed for the first (prefill) step

    # ---- decode generated token ids to text --------------------------------
    generated_tensor = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tensor, skip_special_tokens=False)
    decoded_clean = processor.tokenizer.decode(
        generated_tensor, skip_special_tokens=True
    )
    # #region agent log
    gen_ids = generated_tensor.detach().cpu().tolist()
    _agent_dbg(
        "E",
        "inference.py:test_inference",
        "raw generation",
        {
            "prompt": prompt,
            "decoded": decoded[:500],
            "decoded_clean": decoded_clean[:500],
            "n_gen_tokens": len(gen_ids),
            "gen_ids_head": gen_ids[:32],
        },
    )
    # #endregion

    print(f"\n{'='*60}")
    print(f"Prompt : {prompt}")
    print(f"Output : {decoded_clean}")
    print(f"{'='*60}\n")

    # ---- bounding boxes (<locXXXX>) ----------------------------------------
    loc_matches = re.findall(r"<loc(\d{4})>", decoded)
    locs = []
    if loc_matches:
        locs = [int(x) for x in loc_matches]
        print("[Detection] Bounding boxes (normalised 0-1023):")
        for i in range(0, len(locs) - len(locs) % 4, 4):
            y_min, x_min, y_max, x_max = locs[i : i + 4]
            print(f"  y_min={y_min}, x_min={x_min}, y_max={y_max}, x_max={x_max}")
    # #region agent log
    loc_groups = []
    if locs:
        for i in range(0, len(locs) - len(locs) % 4, 4):
            loc_groups.append(locs[i : i + 4])
    _agent_dbg(
        "B",
        "inference.py:test_inference",
        "parsed loc and seg tokens",
        {
            "loc_matches": loc_matches,
            "locs": locs,
            "n_locs": len(locs),
            "n_loc_groups": len(loc_groups),
            "loc_groups": loc_groups,
            "overlay_uses_first4": locs[:4] if locs else [],
        },
    )
    # #endregion

    # Draw detection bounding boxes overlay
    if loc_groups:
        # Load the original image for drawing bounding boxes
        det_original = Image.open(image_file_path).convert("RGB")
        det_overlaid = draw_bboxes_on_image(det_original, loc_groups, box_color=(255, 0, 0), thickness=3)
        det_path = save_overlay(image_file_path, det_overlaid, suffix="_det")
        print(f"[Detection] Bounding box overlay saved → {det_path}")

    # ---- segmentation mask (<segXXX> or <seqXXX>) --------------------------
    # PaliGemma uses <seg000>…<seg127> for object-centric segmentation.
    seg_matches = re.findall(r"<seg(\d{3})>", decoded)
    if not seg_matches:
        # older checkpoints may use <seq…>
        seg_matches = re.findall(r"<seq(\d{3})>", decoded)

    if seg_matches:
        seg_tokens = [int(x) for x in seg_matches]
        n = len(seg_tokens)
        print(f"[Segmentation] {n} codebook tokens extracted: {seg_tokens}")

        if 1 <= n <= 16:
            if n < 16:
                # Pad with zeros so the VAE always receives a full 4×4 grid.
                # This handles models that emit EOS before all 16 tokens or
                # cases where one token was swallowed by the tokenizer.
                padded = seg_tokens + [0] * (16 - n)
                print(
                    f"[Segmentation] Padded {n} → 16 tokens "
                    f"(appended {16 - n} zero(s)): {padded}"
                )
            else:
                padded = seg_tokens[:16]

            if vae_weights_dir is None or not os.path.isdir(vae_weights_dir):
                print(
                    "[Segmentation] Skipping mask render – provide --vae_weights_dir "
                    "pointing to the vae-oid directory to visualise the mask."
                )
            else:
                print("[Segmentation] Running VAE decoder …")
                mask_np = decode_segmentation_mask(padded, vae_weights_dir, device)

                original_image = Image.open(image_file_path).convert("RGB")

                # Pass bounding box locations (locs) into the overlay function
                overlaid = overlay_mask_on_image(
                    original_image, mask_np, bbox_locs=locs
                )
                out_path = save_overlay(image_file_path, overlaid)

                print(f"[Segmentation] Mask overlay saved → {out_path}")
        else:
            print(f"[Segmentation] Unexpected token count ({n}); skipping mask render.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
    vae_weights_dir: str = None,
) -> None:
    """
    Run PaliGemma inference on a single image+prompt pair.

    Parameters
    ----------
    model_path            : path to the PaliGemma checkpoint directory
    prompt                : task prompt, e.g. "caption en", "detect dog", "segment cat"
    image_file_path       : path to the input image
    max_tokens_to_generate: maximum number of new tokens to generate
    temperature           : sampling temperature (used when do_sample=True)
    top_p                 : nucleus-sampling threshold (used when do_sample=True)
    do_sample             : if True, use nucleus sampling; otherwise greedy
    only_cpu              : if True, force CPU even when a GPU is available
    vae_weights_dir       : directory containing the OID VAE .npy weight files
                            (required for mask visualisation on segmentation tasks)
    """
    # Allow Fire to pass string booleans
    if isinstance(only_cpu, str):
        only_cpu = only_cpu.strip().lower() in ("true", "1", "yes")
    if isinstance(do_sample, str):
        do_sample = do_sample.strip().lower() in ("true", "1", "yes")

    device = "cpu"
    if not only_cpu and torch.cuda.is_available():
        device = "cuda"

    print(f"Device : {device}")

    print("Loading language model …")
    model, tokenizer = load_hf_model(model_path, device)
    model.eval()

    num_image_tokens = model.config.vision_config.num_img_tokens
    image_size = model.config.vision_config.img_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference …")
    with torch.no_grad():
        test_inference(
            model=model,
            processor=processor,
            device=device,
            prompt=prompt,
            image_file_path=image_file_path,
            max_tokens_to_generate=max_tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            vae_weights_dir=vae_weights_dir,
        )


if __name__ == "__main__":
    fire.Fire(main)
