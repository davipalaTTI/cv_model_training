import sys
import os

print("--- SAM 3.1 Environment Diagnostic Test ---\n")

# ==========================================
# 1. GLOBAL IMPORTS (Fixes the red underlines)
# ==========================================
try:
    import torch
    import numpy
    import einops
    import triton
    from PIL import Image
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("✅ All Base Libraries & Meta SAM 3 Imported Successfully.")
except ImportError as e:
    print(f"❌ Missing a required library: {e}")
    print("Please run: pip install Pillow numpy einops triton-windows torch torchvision")
    sys.exit(1)

# ==========================================
# 2. HARDWARE CHECK
# ==========================================
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    device = "cuda:0"
else:
    print("❌ WARNING: PyTorch cannot see your GPU. It will run on CPU.")
    device = "cpu"

# ==========================================
# 3. MODEL LOADING
# ==========================================
print("\n--- Testing Model Initialization ---")
weights_path = os.path.join("weights", "sam3.1_multiplex.pt")

if not os.path.exists(weights_path):
    print(f"❌ Cannot test model loading: '{weights_path}' not found.")
    sys.exit(1)

try:
    print("Loading 3.5GB model into VRAM... (This takes a few seconds)")
    # Build model and push to GPU
    model = build_sam3_image_model(checkpoint_path=weights_path).to(device)

    # Create the Processor (This variable is now globally safe)
    processor = Sam3Processor(model, confidence_threshold=0.25)
    print("✅ SUCCESS! The SAM 3.1 Image Processor successfully loaded.")
except Exception as e:
    print(f"\n❌ CRASH during model load:\n{e}")
    sys.exit(1)

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
print("\n--- Testing Inference Pipeline ---")
test_image_path = r"/\input_raw\0a0a26ce-fe54-4899-bc7f-0d75233527c5_jpg.rf.lDpOoGkiccRIPA96O6aT.jpg"

if not os.path.exists(test_image_path):
    print(f"❌ Test image not found at: {test_image_path}")
    sys.exit(1)

try:
    print("Reading image via PIL...")
    image = Image.open(test_image_path).convert("RGB")

    print("Calculating Image Embeddings... (Using BFloat16 Autocast)")

    # --- THE MAGIC FIX: Wrapping inference in an Autocast block ---
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        inference_state = processor.set_image(image)

        prompt_text = "a license plate"
        print(f"Running text prompt search for '{prompt_text}'...")

        output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

    # --------------------------------------------------------------

    masks = output["masks"]
    boxes = output["boxes"]

    print(f"🎉 COMPLETE SUCCESS! Pipeline ran perfectly and found {len(masks)} potential objects.")
    print("Your environment is completely bulletproof. We can now update inference_sam.py!")

    # ==========================================
    # 6. VISUALIZE THE RESULTS
    # ==========================================
    print("\n--- Generating Visualization ---")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Move PyTorch tensors to CPU memory and convert to Numpy arrays
    masks_np = masks.cpu().numpy()
    boxes_np = boxes.cpu().numpy()

    # 2. Setup the canvas
    plt.figure(figsize=(14, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Define a transparent blue overlay color (RGBA)
    mask_color = np.array([30 / 255, 144 / 255, 255 / 255, 0.5])

    # 3. Loop through every detected object and draw it
    for i, (box, mask) in enumerate(zip(boxes_np, masks_np)):
        # Draw Bounding Box (Format: x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        # Draw Pixel Mask
        if mask.ndim == 3: mask = mask[0]  # Ensure it's 2D
        h, w = mask.shape
        mask_overlay = mask.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
        ax.imshow(mask_overlay)

        # Add a text label
        ax.text(x_min, y_min - 5, f'license plate {i + 1}', color='white', fontsize=10,
                fontweight='bold', backgroundcolor='red')

    plt.title(f"SAM 3.1 Detections for prompt: '{prompt_text}'", fontsize=16)
    plt.axis('off')

    print("🖼️ Popping up image viewer! Close the window to end the script.")
    plt.show()

except Exception as e:
    print(f"\n❌ CRASH during Inference:\n{e}")