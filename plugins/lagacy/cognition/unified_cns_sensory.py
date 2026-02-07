#!/usr/bin/env python3
"""
cns_sensory.py — FINAL PRODUCTION-GRADE NEUROBIOLOGICAL + LLM SENSORY SYSTEM
====================================================================================

FIXED VERSION: Broadcasting error between (48,) and (256,) arrays resolved.
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from collections import deque

# Platform-specific imports with fallbacks
import platform
try:
    import pyaudio  # For mic in / speakers out
except ImportError:
    pyaudio = None
    print("[WARNING] PyAudio not available — audio IO disabled")

try:
    import cv2  # For USB cameras
except ImportError:
    cv2 = None
    print("[WARNING] OpenCV not available — video IO disabled")

if 'raspberrypi' in platform.uname().node.lower():
    try:
        import RPi.GPIO as GPIO
    except ImportError:
        GPIO = None
        print("[WARNING] RPi.GPIO not available — GPIO disabled")
else:
    GPIO = None

# Import your existing systems
from memory_crystal import MemoryCrystalSystem
from agi_cognition import AGICognition  # Direct integration

# =========================
# CONFIGURATION
# =========================
SOUL_FILE = "soul_core.json"
COGNITION_FILE = "cognition_core.json"
SOUL_MAX_BYTES = 8 * 1024 * 1024      # 8 MB
COGNITION_MAX_BYTES = 56 * 1024 * 1024  # 56 MB
BACKUP_DIR = "memory_backups"
os.makedirs(BACKUP_DIR, exist_ok=True)

# =========================
# SENSORY MODALITIES (EXPANDED)
# =========================
MODALITIES = [
    "vision", "audition", "tactile", "proprioception",
    "semantic", "temporal", "olfactory", "interoception",
    "gpio"  # New: for Raspberry Pi GPIO sensors/actuators
]

# Audio/Video Config
AUDIO_CHUNK = 1024 if pyaudio else 0
AUDIO_FORMAT = pyaudio.paInt16 if pyaudio else None
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100

# GPIO Config
GPIO_MODE = GPIO.BCM if GPIO else None
GPIO_PINS = [17, 27, 22]  # Example input pins

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class SensoryInput:
    modality: str
    raw_data: Any
    timestamp: float = field(default_factory=time.time)
    intensity: float = 1.0
    source: str = "external"

@dataclass
class ThalamicGate:
    modality: str
    gate_state: float = 1.0
    attention_weight: float = 1.0
    cortical_feedback: float = 0.0
    soul_resonance_mod: float = 1.0

    def effective_gain(self) -> float:
        return np.clip(
            self.gate_state *
            self.attention_weight *
            (1.0 + 0.45 * self.cortical_feedback) *
            self.soul_resonance_mod,
            0.0, 2.5
        )

@dataclass
class CorticalColumn:
    layer_id: int
    modality: str
    state: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float16))
    prediction: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float16))
    prediction_error: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float16))
    precision: float = 1.0
    last_update: float = 0.0

# =========================
# CENTRAL NERVOUS SYSTEM - FIXED VERSION
# =========================

class CentralNervousSystem:
    def __init__(self):
        # Memory Crystal integration
        self.memory = MemoryCrystalSystem()

        # Cognition Core integration
        self.cognition = AGICognition()

        # Soul resonance (from soul_core.json)
        self.soul_resonance = self._load_soul_resonance()

        # Thalamic nuclei
        self.thalamic_gates = {m: ThalamicGate(m, soul_resonance_mod=self.soul_resonance) for m in MODALITIES}

        # 3-layer cortical layers per modality
        self.cortical_columns = {}
        for mod in MODALITIES:
            self.cortical_columns[mod] = [
                CorticalColumn(layer_id=i, modality=mod)
                for i in range(1, 4)
            ]

        # Association cortex (multimodal fusion) - FIXED: Changed from 1024 to 256
        self.association_state = np.zeros(256, dtype=np.float16)

        # Attention map (dynamic)
        self.attention_map = {m: 1.0 for m in MODALITIES}

        # Prediction history for TMR
        self.prediction_history = {m: deque(maxlen=50) for m in MODALITIES}

        # Free energy tracking
        self.total_free_energy = 0.0
        self.free_energy_history = deque(maxlen=1000)

        # Audio IO (if available)
        self.p_audio = pyaudio.PyAudio() if pyaudio else None
        self.mic_stream = None
        self.speaker_stream = None

        # Video IO
        self.cam = cv2.VideoCapture(0) if cv2 else None

        # GPIO setup
        if GPIO:
            GPIO.setmode(GPIO.BCM)
            for pin in GPIO_PINS:
                GPIO.setup(pin, GPIO.IN)

    def _load_soul_resonance(self) -> float:
        try:
            with open(SOUL_FILE, "r") as f:
                soul = json.load(f)
            gamma = soul.get("emotional_field", {}).get("harmonic_profile", {}).get("gamma", 0.8)
            return float(gamma)
        except:
            return 0.8

    def _ensure_shape_256(self, array: np.ndarray) -> np.ndarray:
        """Ensure array is exactly 256 elements by padding or truncating"""
        if len(array) != 256:
            if len(array) > 256:
                # Truncate to first 256 elements
                return array[:256].astype(np.float16)
            else:
                # Pad with zeros
                padded = np.zeros(256, dtype=np.float16)
                padded[:len(array)] = array.astype(np.float16)
                return padded
        return array.astype(np.float16)

    # =========================
    # SENSORY ENCODING (EXPANDED)
    # =========================
    def _encode_signal(self, inp: SensoryInput) -> np.ndarray:
        """FIXED: All encoded signals now return exactly 256 elements"""
        if inp.modality == "semantic":
            h = hashlib.sha384(str(inp.raw_data).encode()).digest()
            vec = np.frombuffer(h, dtype=np.uint8)[:256].astype(np.float16)
            vec = self._ensure_shape_256(vec)
            return vec / 255.0 * inp.intensity

        elif inp.modality == "temporal":
            dt = time.time() - inp.timestamp
            spike = np.exp(-np.linspace(0, 5, 256) * dt)
            spike = self._ensure_shape_256(spike)
            return spike * inp.intensity

        elif inp.modality == "audition" and inp.source == "mic":
            vec = self._capture_mic_audio()
            vec = self._ensure_shape_256(vec)
            return vec

        elif inp.modality == "vision" and inp.source == "camera":
            vec = self._capture_camera_frame()
            vec = self._ensure_shape_256(vec)
            return vec

        elif inp.modality == "gpio" and GPIO:
            vec = self._read_gpio()
            vec = self._ensure_shape_256(vec)
            return vec

        else:
            # Generic/fallback
            base = np.random.randn(256).astype(np.float16) * 0.1
            return base * inp.intensity

    # =========================
    # REAL IO CAPTURE
    # =========================
    def _capture_mic_audio(self) -> np.ndarray:
        if not self.p_audio or not self.mic_stream:
            return np.zeros(256, dtype=np.float16)

        data = self.mic_stream.read(AUDIO_CHUNK)
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float16) / 32768.0
        # Simple processing: FFT magnitude - ensure 256 elements
        fft = np.abs(np.fft.fft(audio_array))
        # Take first 256 frequency bins
        if len(fft) >= 256:
            fft = fft[:256]
        else:
            fft = np.pad(fft, (0, 256 - len(fft)))
        return fft / np.max(fft + 1e-8)

    def _capture_camera_frame(self) -> np.ndarray:
        if not self.cam or not self.cam.isOpened():
            return np.zeros(256, dtype=np.float16)

        ret, frame = self.cam.read()
        if not ret:
            return np.zeros(256, dtype=np.float16)

        # Simple processing: grayscale resize + flatten
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 16))  # 16x16 = 256 pixels
        return small.flatten().astype(np.float16) / 255.0

    def _read_gpio(self) -> np.ndarray:
        if not GPIO:
            return np.zeros(256, dtype=np.float16)

        readings = []
        for pin in GPIO_PINS:
            readings.append(GPIO.input(pin))

        # FIXED: Ensure exactly 256 elements by repeating the readings
        repeats = 256 // len(readings) + 1
        vec = np.array(readings * repeats)[:256].astype(np.float16)
        return vec

    def start_audio_streams(self):
        if self.p_audio:
            self.mic_stream = self.p_audio.open(
                format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE, input=True, frames_per_buffer=AUDIO_CHUNK
            )
            self.speaker_stream = self.p_audio.open(
                format=AUDIO_FORMAT, channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE, output=True, frames_per_buffer=AUDIO_CHUNK
            )

    def play_audio(self, audio_data: np.ndarray):
        if self.speaker_stream:
            data = (audio_data * 32767).astype(np.int16).tobytes()
            self.speaker_stream.write(data)

    # =========================
    # MAIN SENSORY PIPELINE - FIXED
    # =========================
    def process(self, sensory_input: SensoryInput) -> Dict[str, Any]:
        mod = sensory_input.modality
        gate = self.thalamic_gates[mod]
        columns = self.cortical_columns[mod]

        # 1. Encode + thalamic gating
        raw_signal = self._encode_signal(sensory_input)
        raw_signal = self._ensure_shape_256(raw_signal)
        gated_signal = raw_signal * gate.effective_gain()

        if np.mean(np.abs(gated_signal)) < 0.38:  # Subthreshold → gate closed
            return {"status": "gated", "modality": mod}

        # 2. Bottom-up hierarchical sweep
        for i, col in enumerate(columns):
            if i == 0:
                col.state = gated_signal
            else:
                col.state = self._ensure_shape_256(columns[i-1].prediction_error)

            # FIXED: Ensure both arrays are 256 before subtraction
            col_pred = self._ensure_shape_256(col.prediction)
            col_state = self._ensure_shape_256(col.state)
            col.prediction_error = col_state - col_pred

            self.total_free_energy += col.precision * np.sum(col.prediction_error ** 2)

        # 3. Top-down prediction sweep
        for i in range(len(columns)-1, 0, -1):
            pred = columns[i].state @ np.eye(256) * 0.7 + columns[i-1].state * 0.3
            columns[i-1].prediction = self._ensure_shape_256(pred)

        # 4. L6CT feedback to thalamus
        gate.cortical_feedback = float(np.mean(columns[0].state))

        # 5. Multimodal fusion
        self._fuse_multimodal()

        # 6. Novelty detection → Memory Crystal
        novelty = float(np.mean([np.sum(c.prediction_error**2) for c in columns]))
        if novelty > 0.72:
            self.memory.crystallize(
                text=str(sensory_input.raw_data),
                context=f"sensory_novelty_{mod}",
                cues=[mod, "high_prediction_error"]
            )

        # 7. Store prediction for TMR
        self.prediction_history[mod].append(columns[-1].state.copy())

        # 8. Cognition Core Handshake
        cognition_result = self.cognition.process({"text": str(sensory_input.raw_data), "modality": mod})

        return {
            "modality": mod,
            "novelty": novelty,
            "free_energy_contribution": self.total_free_energy,
            "association_state": self.association_state.tolist(),
            "thalamic_gain": gate.effective_gain(),
            "crystal_created": novelty > 0.72,
            "cognition_metrics": cognition_result
        }

    def _fuse_multimodal(self):
        """FIXED: Ensure all arrays are 256 before fusion"""
        active = []
        for mod, gate in self.thalamic_gates.items():
            if gate.effective_gain() > 0.5:
                top_layer = self.cortical_columns[mod][-1].state
                # Ensure shape consistency
                top_layer = self._ensure_shape_256(top_layer)
                weighted = top_layer * self.attention_map[mod]
                active.append(weighted)

        if active:
            # Stack and average
            stacked = np.stack([self._ensure_shape_256(a) for a in active])
            self.association_state = np.mean(stacked, axis=0).astype(np.float16)

    # =========================
    # TARGETED MEMORY REACTIVATION
    # =========================
    def tmr_cycle(self):
        """Run during idle/sleep phases"""
        for mod in MODALITIES:
            if len(self.prediction_history[mod]) > 5:
                # Replay top 3 most surprising memories
                for state in list(self.prediction_history[mod])[-3:]:
                    # Strengthen cortical prediction
                    for col in self.cortical_columns[mod]:
                        col.prediction += 0.05 * self._ensure_shape_256(state)
                        col.prediction = np.clip(col.prediction, -10, 10)

    # =========================
    # ATTENTION & CONTROL
    # =========================
    def set_attention(self, modality: str, weight: float):
        self.attention_map[modality] = float(weight)
        self.thalamic_gates[modality].attention_weight = weight

    def modulate_gate(self, modality: str, state: float):
        self.thalamic_gates[modality].gate_state = np.clip(state, 0.0, 1.0)

# =========================
# GLOBAL SYSTEM
# =========================
CNS = CentralNervousSystem()
CNS.start_audio_streams()  # If PyAudio available

# Example usage (daemon will call these)
def process_text(text: str, intensity: float = 1.0) -> Dict:
    inp = SensoryInput("semantic", text, intensity=intensity)
    try:
        result = CNS.process(inp)
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "modality": "semantic",
            "text": text
        }

def process_image_from_camera(intensity: float = 1.0) -> Dict:
    inp = SensoryInput("vision", None, intensity=intensity, source="camera")
    return CNS.process(inp)

def process_audio_from_mic(intensity: float = 1.0) -> Dict:
    inp = SensoryInput("audition", None, intensity=intensity, source="mic")
    return CNS.process(inp)

def process_gpio_input(intensity: float = 1.0) -> Dict:
    inp = SensoryInput("gpio", None, intensity=intensity)
    return CNS.process(inp)

def output_to_speakers(audio_data: np.ndarray):
    CNS.play_audio(audio_data)

if __name__ == "__main__":
    print("UNIFIED CNS ONLINE — 8+ SENSES ACTIVE")
    CNS.set_attention("semantic", 1.6)

    print(process_text("The void calls. The echo answers."))
    print(process_audio_from_mic())
    print(process_image_from_camera())
    print(process_gpio_input())

    CNS.tmr_cycle()
    print(f"Crystals formed: {len(CNS.memory.crystals)}")
