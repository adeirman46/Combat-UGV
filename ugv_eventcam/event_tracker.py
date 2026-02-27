#!/usr/bin/env python3
"""
SNN-Based Event Camera Tracker (v2 — spikingjelly + Pre-trained Model)
======================================================================
Uses spikingjelly framework with a pre-trained CIFAR10 ResNet18 SNN backbone
(downloaded from GitHub/figshare) as a frozen feature extractor, plus a
lightweight PLIF (Parametric Leaky Integrate-and-Fire) detection head
for real-time webcam event-based object tracking.

Pre-trained model: SJ-cifar10-resnet18_model-sample.pth (~45MB)
Source: https://github.com/fangwei123456/spikingjelly

4-Panel Visualization:
  Top-Left:     Original BGR feed with bounding box + speed
  Top-Right:    Event polarity map (Red=ON, Blue=OFF)
  Bottom-Left:  SNN spike activation heatmap
  Bottom-Right: Membrane potential visualization
"""

import cv2
import numpy as np
import time
import math
import os
import urllib.request
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── spikingjelly imports ──
from spikingjelly.activation_based import neuron, layer, functional, surrogate

# =============================================================================
# Pre-trained Model Management
# =============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "SJ-cifar10-resnet18_model-sample.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://ndownloader.figshare.com/files/26676110"


def download_model():
    """Download the pre-trained spikingjelly CIFAR10 ResNet18 SNN model."""
    if os.path.exists(MODEL_PATH):
        print(f"[Model] Found pre-trained model: {MODEL_PATH}")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[Model] Downloading pre-trained SNN from GitHub/figshare...")
    print(f"[Model] URL: {MODEL_URL}")
    print(f"[Model] Target: {MODEL_PATH}")

    try:
        def _progress(count, block_size, total_size):
            pct = count * block_size * 100 / total_size
            sys.stdout.write(f"\r[Model] Downloading... {pct:.1f}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
        print(f"\n[Model] Download complete ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"\n[Model] Download failed: {e}")
        print("[Model] Continuing with randomly initialized backbone...")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)


# =============================================================================
# Pre-trained Feature Backbone (frozen ResNet18 SNN early layers)
# =============================================================================
class PretrainedBackbone(nn.Module):
    """
    Loads early convolutional layers from the pre-trained CIFAR10 ResNet18 SNN
    and uses them as a frozen feature extractor.

    The pre-trained model expects 3-channel input. We adapt our 2-channel
    event data (ON/OFF) by replicating/padding to 3 channels.
    """

    def __init__(self, device):
        super().__init__()

        # Build a small conv feature extractor that mirrors ResNet18 early layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )

        # Try to load pre-trained weights
        self._load_pretrained(device)

        # Freeze all parameters — this backbone is read-only
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

    def _load_pretrained(self, device):
        """Load matching weights from the pre-trained checkpoint."""
        if not os.path.exists(MODEL_PATH):
            print("[Backbone] No pre-trained weights found, using random init")
            return

        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

            # The checkpoint may be a state_dict or contain one
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Find and load matching conv1 weights
            loaded = 0
            for name, param in state_dict.items():
                # Look for first conv layer weights (64 output channels, 3 input)
                if 'conv' in name.lower() and param.shape == torch.Size([64, 3, 3, 3]):
                    self.features[0].weight.data.copy_(param)
                    loaded += 1
                    print(f"[Backbone] Loaded conv weights from: {name} {list(param.shape)}")
                    break
                # Also check for plain weight tensors
                if param.dim() == 4 and param.shape[0] == 64 and param.shape[1] == 3:
                    self.features[0].weight.data.copy_(param)
                    loaded += 1
                    print(f"[Backbone] Loaded conv weights from: {name} {list(param.shape)}")
                    break

            # Find and load BatchNorm weights
            for name, param in state_dict.items():
                if 'bn' in name.lower() and 'weight' in name.lower() and param.shape == torch.Size([64]):
                    self.features[1].weight.data.copy_(param)
                    loaded += 1
                    print(f"[Backbone] Loaded BN weight from: {name}")
                    break

            for name, param in state_dict.items():
                if 'bn' in name.lower() and 'bias' in name.lower() and param.shape == torch.Size([64]):
                    self.features[1].bias.data.copy_(param)
                    loaded += 1
                    print(f"[Backbone] Loaded BN bias from: {name}")
                    break

            if loaded > 0:
                print(f"[Backbone] Successfully loaded {loaded} parameter groups from pre-trained model")
            else:
                print("[Backbone] Could not match weights, using random init")
                # Print the available keys for debugging
                print(f"[Backbone] Available checkpoint keys ({len(state_dict)}):")
                for i, (k, v) in enumerate(state_dict.items()):
                    if i < 10:
                        print(f"  {k}: {list(v.shape) if hasattr(v, 'shape') else type(v)}")

        except Exception as e:
            print(f"[Backbone] Error loading pre-trained weights: {e}")
            print("[Backbone] Using random initialization")

    def forward(self, x):
        """
        x: (B, 2, H, W) — ON/OFF event channels
        Returns: (B, 64, H/2, W/2) — feature maps
        """
        # Adapt 2-channel events to 3-channel input expected by pre-trained model
        # Channel 0 = ON events, Channel 1 = OFF events, Channel 2 = combined magnitude
        combined = torch.clamp(x[:, 0:1] + x[:, 1:2], 0, 1)
        x3ch = torch.cat([x, combined], dim=1)  # (B, 3, H, W)

        return self.features(x3ch)


# =============================================================================
# Lightweight SNN Detection Head (PLIF neurons)
# =============================================================================
class SpikeDetectionHead(nn.Module):
    """
    Lightweight detection head using spikingjelly PLIF neurons.
    Takes backbone features and produces a spike activation map for tracking.

    Architecture:
        Input: 64-channel feature maps from backbone
        Conv1: 64 -> 32 channels, 3x3 + PLIF neuron
        Conv2: 32 -> 16 channels, 3x3 + PLIF neuron
        Conv3: 16 -> 1 channel,  1x1 + PLIF neuron (output activation map)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.plif1 = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.plif2 = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

        self.conv3 = nn.Conv2d(16, 1, kernel_size=1, bias=False)
        self.plif3 = neuron.ParametricLIFNode(
            init_tau=3.0,
            v_threshold=0.3,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

    def forward(self, x):
        """
        x: (B, 64, H', W') from backbone
        Returns:
            spk: (B, 1, H', W') spike output
            mem: (B, 1, H', W') membrane potential (from last PLIF)
        """
        # Layer 1
        x = self.bn1(self.conv1(x))
        x = self.plif1(x)

        # Layer 2
        x = self.bn2(self.conv2(x))
        x = self.plif2(x)

        # Layer 3 — output
        x = self.conv3(x)

        # Save pre-spike membrane potential for visualization
        mem = x.clone()
        spk = self.plif3(x)

        return spk, mem


# =============================================================================
# Complete SNN Event Tracker Model
# =============================================================================
class EventSNNTracker(nn.Module):
    """
    Full pipeline: PretrainedBackbone → SpikeDetectionHead

    Uses pre-trained CIFAR10 ResNet18 SNN backbone (frozen)
    with a lightweight PLIF detection head for event-based tracking.
    """

    def __init__(self, device):
        super().__init__()
        self.backbone = PretrainedBackbone(device)
        self.head = SpikeDetectionHead()

    def forward(self, x):
        """
        x: (B, 2, H, W) — ON/OFF event tensor
        Returns: spk, mem from detection head
        """
        features = self.backbone(x)
        return self.head(features)




# =============================================================================
# Visualization Helpers
# =============================================================================
def make_spike_heatmap(spike_data, target_shape):
    """Convert spike data to a colorized heatmap."""
    spike_np = spike_data.squeeze().cpu().numpy()
    max_val = spike_np.max()
    if max_val > 0:
        spike_norm = (spike_np / max_val * 255).astype(np.uint8)
    else:
        spike_norm = np.zeros_like(spike_np, dtype=np.uint8)
    resized = cv2.resize(spike_norm, (target_shape[1], target_shape[0]))
    return cv2.applyColorMap(resized, cv2.COLORMAP_INFERNO)


def make_membrane_vis(membrane, target_shape):
    """Visualize membrane potential as a colormap."""
    mem_np = membrane.squeeze().cpu().detach().numpy()
    abs_mem = np.abs(mem_np)
    max_val = abs_mem.max()
    if max_val > 0:
        mem_norm = (abs_mem / max_val * 255).astype(np.uint8)
    else:
        mem_norm = np.zeros_like(abs_mem, dtype=np.uint8)
    resized = cv2.resize(mem_norm, (target_shape[1], target_shape[0]))
    return cv2.applyColorMap(resized, cv2.COLORMAP_VIRIDIS)


def draw_label(img, text, pos, color=(0, 255, 0), scale=0.55, thickness=2):
    """Draw text with a dark background for readability."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# =============================================================================
# Main Loop
# =============================================================================
def main():
    print("=" * 60)
    print("  SNN Event Camera Tracker v2")
    print("  Framework: spikingjelly (github.com/fangwei123456/spikingjelly)")
    print("  Model: Pre-trained CIFAR10 ResNet18 SNN backbone")
    print("=" * 60)

    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # ── Download pre-trained model ──
    download_model()

    # ── Webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        return

    h, w = prev_frame.shape[:2]
    print(f"[Camera] Resolution: {w}x{h}")

    # ── Previous frame prep ──
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0)
    prev_gray_f = prev_gray.astype(np.float32)

    # ── Build SNN Model ──
    print("[SNN] Building model...")
    model = EventSNNTracker(device).to(device)
    model.eval()

    # Initialize detection head with Kaiming
    for m in model.head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data *= 2.5  # scale up so events trigger spikes

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[SNN] Total params: {total_params:,}")
    print(f"[SNN] Frozen (backbone): {frozen_params:,}")
    print(f"[SNN] Trainable (head): {trainable_params:,}")
    print()
    print(model)
    print()

    # ── Tracking variables ──
    EVENT_THRESHOLD = 20
    METERS_PER_PIXEL = 0.01
    prev_center = None
    prev_time = time.time()
    smoothed_fps = 0.0
    smoothed_speed_kmh = 0.0

    # Spike accumulator
    ACCUM_FRAMES = 4
    spike_accum = None
    frame_count = 0

    print("[SNN] Ready! Press 'q' to quit.")
    print("-" * 60)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            current_fps = 1.0 / dt if dt > 0 else 0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * current_fps

            # ── Generate Events ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)
            gray_f = gray.astype(np.float32)

            diff = gray_f - prev_gray_f
            on_events = (diff > EVENT_THRESHOLD).astype(np.uint8) * 255
            off_events = (diff < -EVENT_THRESHOLD).astype(np.uint8) * 255
            prev_gray_f = gray_f

            # ── Event visualization (Top-Right) ──
            event_vis = np.zeros_like(frame)
            event_vis[on_events == 255] = [0, 0, 255]    # Red = ON
            event_vis[off_events == 255] = [255, 0, 0]   # Blue = OFF

            # ── Create event tensor ──
            on_t = torch.from_numpy(on_events.astype(np.float32) / 255.0)
            off_t = torch.from_numpy(off_events.astype(np.float32) / 255.0)
            event_tensor = torch.stack([on_t, off_t], dim=0).unsqueeze(0).to(device)

            # ── SNN Forward Pass ──
            functional.reset_net(model)  # reset neuron states between frames

            # Run multiple timesteps for temporal integration
            NUM_TIMESTEPS = 4
            spk_sum = None
            last_mem = None

            for t in range(NUM_TIMESTEPS):
                # Slightly vary the input to simulate temporal dynamics
                noise_scale = 1.0 - 0.1 * t
                spk_out, mem_out = model(event_tensor * noise_scale)

                if spk_sum is None:
                    spk_sum = spk_out.clone()
                else:
                    spk_sum += spk_out
                last_mem = mem_out

            # Accumulate across frames
            if spike_accum is None:
                spike_accum = spk_sum.clone()
            else:
                spike_accum += spk_sum
            frame_count += 1

            detection_map = spike_accum.clone()
            if frame_count >= ACCUM_FRAMES:
                spike_accum = None
                frame_count = 0

            # ── Object detection from SNN output ──
            det_np = detection_map.squeeze().cpu().numpy()
            # Upsample detection map to original resolution
            det_resized = cv2.resize(det_np, (w, h))
            max_val = det_resized.max()
            if max_val > 0:
                det_norm = (det_resized / max_val * 255).astype(np.uint8)
            else:
                det_norm = np.zeros((h, w), dtype=np.uint8)

            _, det_thresh = cv2.threshold(det_norm, 40, 255, cv2.THRESH_BINARY)
            det_thresh = cv2.dilate(det_thresh, None, iterations=3)
            det_thresh = cv2.erode(det_thresh, None, iterations=1)

            contours, _ = cv2.findContours(
                det_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # ── Tracking ──
            object_tracked = False
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 400:
                    object_tracked = True
                    x, y, bw, bh = cv2.boundingRect(largest)
                    cx, cy = x + bw // 2, y + bh // 2

                    # Bounding box
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.circle(event_vis, (cx, cy), 8, (0, 255, 0), -1)

                    # Speed
                    if prev_center is not None and dt > 0:
                        dx = cx - prev_center[0]
                        dy = cy - prev_center[1]
                        dist_px = math.hypot(dx, dy)
                        speed_mps = (dist_px * METERS_PER_PIXEL) / dt
                        speed_kmh = speed_mps * 3.6
                        smoothed_speed_kmh = 0.8 * smoothed_speed_kmh + 0.2 * speed_kmh

                    prev_center = (cx, cy)

                    draw_label(frame, f"Track: ({cx},{cy})", (x, y - 35), (0, 255, 0))
                    draw_label(frame, f"Speed: {smoothed_speed_kmh:.1f} km/h",
                               (x, y - 10), (0, 255, 0))

            if not object_tracked:
                prev_center = None

            # ── Visualization panels ──
            spike_heatmap = make_spike_heatmap(detection_map, (h, w))
            membrane_vis = make_membrane_vis(last_mem, (h, w))

            # HUD labels
            draw_label(frame, f"FPS: {int(smoothed_fps)}", (10, 28), (0, 255, 0))
            draw_label(frame, f"Device: {device}", (10, 56), (200, 200, 200), 0.45, 1)
            draw_label(frame, "spikingjelly PLIF", (10, 78), (0, 200, 255), 0.45, 1)

            draw_label(event_vis, "Events (Red=ON, Blue=OFF)", (10, 28), (255, 255, 255))

            total_spikes = int(spk_sum.sum().item())
            mean_mem = float(last_mem.mean().item())
            draw_label(spike_heatmap, "SNN Spike Activation", (10, 28), (255, 255, 255))
            draw_label(spike_heatmap, f"Spikes/frame: {total_spikes}", (10, 56),
                       (0, 200, 255), 0.45, 1)
            draw_label(membrane_vis, "Membrane Potential (PLIF)", (10, 28), (255, 255, 255))
            draw_label(membrane_vis, f"Mean Vmem: {mean_mem:.4f}", (10, 56),
                       (0, 200, 255), 0.45, 1)

            # ── Compose 4-panel ──
            top_row = np.hstack((frame, event_vis))
            bot_row = np.hstack((spike_heatmap, membrane_vis))
            combined = np.vstack((top_row, bot_row))

            cv2.imshow("SNN Event Camera Tracker v2 (spikingjelly)", combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[SNN Event Tracker] Stopped.")


if __name__ == "__main__":
    main()
