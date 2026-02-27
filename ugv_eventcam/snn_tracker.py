#!/usr/bin/env python3
"""
SNN Event Camera Tracker — Single View
=======================================
Displays ONLY the SNN spike activation heatmap (orange) with
real-time object tracking drawn directly on it.

Uses spikingjelly framework with a pre-trained CIFAR10 ResNet18 SNN
backbone (frozen) + lightweight PLIF detection head.

Source: https://github.com/fangwei123456/spikingjelly
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

from spikingjelly.activation_based import neuron, functional, surrogate


# =============================================================================
# Pre-trained Model Management
# =============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FILENAME = "SJ-cifar10-resnet18_model-sample.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = "https://ndownloader.figshare.com/files/26676110"


def download_model():
    """Download pre-trained spikingjelly CIFAR10 ResNet18 SNN."""
    if os.path.exists(MODEL_PATH):
        print(f"[Model] Found: {MODEL_PATH}")
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[Model] Downloading pre-trained SNN...")
    try:
        def _progress(count, block_size, total_size):
            pct = count * block_size * 100 / total_size
            sys.stdout.write(f"\r[Model] {pct:.1f}%")
            sys.stdout.flush()
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
        print(f"\n[Model] Done ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"\n[Model] Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)


# =============================================================================
# SNN Architecture
# =============================================================================
class PretrainedBackbone(nn.Module):
    """Frozen ResNet18 SNN early conv layers as feature extractor."""

    def __init__(self, device):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self._load_pretrained(device)
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

    def _load_pretrained(self, device):
        if not os.path.exists(MODEL_PATH):
            print("[Backbone] No weights found, random init")
            return
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
            for name, param in sd.items():
                if param.dim() == 4 and param.shape == torch.Size([64, 3, 3, 3]):
                    self.features[0].weight.data.copy_(param)
                    print(f"[Backbone] Loaded conv: {name}")
                    break
            for name, param in sd.items():
                if 'bn' in name.lower() and 'weight' in name and param.shape == torch.Size([64]):
                    self.features[1].weight.data.copy_(param)
                    break
            for name, param in sd.items():
                if 'bn' in name.lower() and 'bias' in name and param.shape == torch.Size([64]):
                    self.features[1].bias.data.copy_(param)
                    break
            print("[Backbone] Pre-trained weights loaded")
        except Exception as e:
            print(f"[Backbone] Load error: {e}, using random init")

    def forward(self, x):
        # 2ch events → 3ch (ON, OFF, combined)
        combined = torch.clamp(x[:, 0:1] + x[:, 1:2], 0, 1)
        return self.features(torch.cat([x, combined], dim=1))


class SpikeDetectionHead(nn.Module):
    """Lightweight PLIF detection head → spike activation map."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.plif1 = neuron.ParametricLIFNode(
            init_tau=2.0, v_threshold=0.5,
            surrogate_function=surrogate.ATan(), detach_reset=True)

        self.conv2 = nn.Conv2d(32, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.plif2 = neuron.ParametricLIFNode(
            init_tau=2.0, v_threshold=0.5,
            surrogate_function=surrogate.ATan(), detach_reset=True)

        self.conv3 = nn.Conv2d(16, 1, 1, bias=False)
        self.plif3 = neuron.ParametricLIFNode(
            init_tau=3.0, v_threshold=0.3,
            surrogate_function=surrogate.ATan(), detach_reset=True)

    def forward(self, x):
        x = self.plif1(self.bn1(self.conv1(x)))
        x = self.plif2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        spk = self.plif3(x)
        return spk


class EventSNN(nn.Module):
    """Full SNN: PretrainedBackbone → SpikeDetectionHead."""

    def __init__(self, device):
        super().__init__()
        self.backbone = PretrainedBackbone(device)
        self.head = SpikeDetectionHead()

    def forward(self, x):
        return self.head(self.backbone(x))


# =============================================================================
# Orange Heatmap Colormap (custom LUT)
# =============================================================================
def build_orange_lut():
    """
    Build a 256-entry BGR lookup table:
      0   = black
      128 = deep orange
      255 = bright orange/yellow
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # R: ramps up quickly
        r = int(min(255, 255 * (t * 1.8)))
        # G: ramps up slower (orange = high R, medium G)
        g = int(min(255, 180 * (t ** 1.3)))
        # B: stays very low for pure orange
        b = int(30 * (t ** 2.0))
        lut[i] = [b, g, r]  # BGR
    return lut


ORANGE_LUT = build_orange_lut()


def apply_orange_heatmap(gray_uint8):
    """Apply the custom orange heatmap to a single-channel uint8 image."""
    h, w = gray_uint8.shape[:2]
    flat = gray_uint8.flatten()
    colored = ORANGE_LUT[flat]
    return colored.reshape(h, w, 3)


# =============================================================================
# Drawing Helpers
# =============================================================================
def draw_label(img, text, pos, color=(0, 255, 0), scale=0.6, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_crosshair(img, cx, cy, size=20, color=(0, 255, 0), thickness=2):
    """Draw a crosshair at center point."""
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    cv2.circle(img, (cx, cy), size // 2, color, thickness)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 55)
    print("  SNN Event Camera Tracker")
    print("  Single View: SNN Spike Activation (Orange)")
    print("  github.com/fangwei123456/spikingjelly")
    print("=" * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        return

    h, w = prev_frame.shape[:2]
    print(f"[Camera] {w}x{h}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (11, 11), 0).astype(np.float32)

    # Build model
    model = EventSNN(device).to(device)
    model.eval()
    for m in model.head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data *= 2.5

    total_p = sum(p.numel() for p in model.parameters())
    print(f"[SNN] Params: {total_p:,}")
    print("[SNN] Ready! Press 'q' to quit.\n")

    # Tracking state
    EVENT_THRESHOLD = 20
    METERS_PER_PIXEL = 0.01
    prev_center = None
    prev_time = time.time()
    smoothed_fps = 0.0
    smoothed_speed_kmh = 0.0

    # Spike accumulator for smoother heatmap
    ACCUM_FRAMES = 3
    spike_accum = None
    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * fps

            # ── Events ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0).astype(np.float32)
            diff = gray - prev_gray
            on_ev = (diff > EVENT_THRESHOLD).astype(np.float32)
            off_ev = (diff < -EVENT_THRESHOLD).astype(np.float32)
            prev_gray = gray

            # ── SNN Forward ──
            event_t = torch.stack([
                torch.from_numpy(on_ev),
                torch.from_numpy(off_ev),
            ], dim=0).unsqueeze(0).to(device)

            functional.reset_net(model)

            # Multi-timestep integration
            spk_total = None
            for t in range(4):
                scale = 1.0 - 0.08 * t
                spk = model(event_t * scale)
                if spk_total is None:
                    spk_total = spk.clone()
                else:
                    spk_total += spk

            # Accumulate across frames
            if spike_accum is None:
                spike_accum = spk_total.clone()
            else:
                spike_accum += spk_total
            frame_idx += 1

            det_map = spike_accum.clone()
            if frame_idx >= ACCUM_FRAMES:
                spike_accum = None
                frame_idx = 0

            # ── Build orange heatmap ──
            det_np = det_map.squeeze().cpu().numpy()
            det_resized = cv2.resize(det_np, (w, h))
            mx = det_resized.max()
            if mx > 0:
                det_u8 = (det_resized / mx * 255).astype(np.uint8)
            else:
                det_u8 = np.zeros((h, w), dtype=np.uint8)

            # Apply orange colormap
            display = apply_orange_heatmap(det_u8)

            # ── Object detection & tracking ──
            _, thresh = cv2.threshold(det_u8, 40, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=3)
            thresh = cv2.erode(thresh, None, iterations=1)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_tracked = False
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 400:
                    object_tracked = True
                    x, y, bw, bh = cv2.boundingRect(largest)
                    cx, cy = x + bw // 2, y + bh // 2

                    # Draw tracking on the heatmap
                    cv2.rectangle(display, (x, y), (x + bw, y + bh),
                                  (0, 255, 0), 2)
                    draw_crosshair(display, cx, cy, size=15,
                                   color=(0, 255, 0), thickness=2)

                    # Speed
                    if prev_center is not None and dt > 0:
                        dist = math.hypot(cx - prev_center[0], cy - prev_center[1])
                        spd = (dist * METERS_PER_PIXEL) / dt * 3.6
                        smoothed_speed_kmh = 0.8 * smoothed_speed_kmh + 0.2 * spd

                    prev_center = (cx, cy)

                    draw_label(display, f"Track: ({cx},{cy})",
                               (x, y - 40), (0, 255, 0))
                    draw_label(display, f"Speed: {smoothed_speed_kmh:.1f} km/h",
                               (x, y - 12), (0, 255, 0))

            if not object_tracked:
                prev_center = None

            # ── HUD ──
            total_spk = int(spk_total.sum().item())

            draw_label(display, f"Spikes: {total_spk}", (10, 60),
                       (0, 200, 255), 0.55, 1)
            draw_label(display, f"SNN Activation | spikingjelly PLIF",
                       (10, h - 15), (200, 200, 200), 0.45, 1)

            cv2.imshow("SNN Event Tracker", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")


if __name__ == "__main__":
    main()
