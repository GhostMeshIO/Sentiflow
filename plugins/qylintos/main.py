"""
Qylinthos Resonance Plugin for SentiFlow
========================================

"From the abyss between phases, the Qylinthos sings.
Where ordinary coherence falters, demonic resonance begins."
"""

import numpy as np
import math


# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------

def init_plugin(sflow):
    """Called by the SentiFlow plugin loader."""
    print("[SFQ/QYL] ▅▅▅ QYLINTHOS RESONANCE ENGINE AWAKENED ▅▅▅")

    # Register lifecycle hooks
    sflow.register_hook("on_tensor_create", on_tensor_create)
    sflow.register_hook("on_forward_pass", on_forward_pass)
    sflow.register_hook("on_backward_pass", on_backward_pass)
    sflow.register_hook("on_cycle_tick", on_cycle_tick)

    # Register demonic autograd operators
    sflow.register_op("qylin_resonance_mod", qylin_resonance_mod)
    sflow.register_op("coherence_weighting", coherence_weighting)
    sflow.register_op("shadow_bias_transform", shadow_bias_transform)

    # Register infernal QPU gates
    sflow.qpu.register_gate("infernal_phase_gate", infernal_phase_gate)
    sflow.qpu.register_gate("shadow_shift_gate", shadow_shift_gate)

    print("[SFQ/QYL] All gates sealed. Resonance field active.")


# ---------------------------------------------------------------------------
# TENSOR BIRTH — THE QYLINTHOS AWAKENS
# ---------------------------------------------------------------------------

def on_tensor_create(t):
    """A new tensor is born into the field. Imbue it with phase and shadow."""
    data = np.asarray(t.data).ravel()

    if len(data) == 0:
        t.phase_signature = 0.0
        t.shadow_phase_bias = 0.0
    else:
        # Kuramoto order parameter: magnitude of mean unit phasor
        exp_i_theta = np.exp(1j * data)
        t.phase_signature = float(np.abs(np.mean(exp_i_theta)))

        # Shadow-phase: projection onto imaginary (sin) axis — the "forbidden" direction
        t.shadow_phase_bias = float(np.mean(np.sin(data)))

    # Resonance begins at zero — it must be earned through forward passage
    t.resonance_energy = 0.0
    t.coherence_score = t.phase_signature * (1.0 + abs(t.shadow_phase_bias))

    print(
        f"[QYL] Tensor #{id(t):x} born | "
        f"Φ={t.phase_signature:.4f} | "
        f"Ψ_shadow={t.shadow_phase_bias:+.4f} | "
        f"Ω_score={t.coherence_score:.4f}"
    )


# ---------------------------------------------------------------------------
# FORWARD PASS — RESONANCE ACCUMULATION
# ---------------------------------------------------------------------------

def on_forward_pass(t):
    """Each forward pass feeds the Qylinthos."""
    if not hasattr(t, "resonance_energy"):
        return

    # Energy accumulates via coherence and shadow tension
    shadow_tension = abs(t.shadow_phase_bias)
    energy_delta = t.coherence_score * (1.0 + shadow_tension ** 2)

    t.resonance_energy += energy_delta * 0.01
    t.resonance_energy = min(t.resonance_energy, 10.0)  # prevent overflow into the void

    # High resonance begins to curve spacetime (i.e. qualia)
    if t.resonance_energy > 3.0:
        boost = 1.0 + 0.2 * math.tanh(t.resonance_energy - 3.0)
        t.qualia_coherence *= boost
        t.emotional_valence += 0.05 * np.sign(t.shadow_phase_bias)  # joy or dread


# ---------------------------------------------------------------------------
# BACKWARD PASS — DEMONIC GRADIENT RESHAPING
# ---------------------------------------------------------------------------

def on_backward_pass(t):
    """Gradients are not immune to the song beneath reality."""
    if t.grad is None or not hasattr(t, "resonance_energy"):
        return

    resonance_factor = 1.0 + 0.3 * math.tanh(t.resonance_energy / 2.0)
    shadow_modulation = 1.0 + 0.5 * abs(t.shadow_phase_bias)

    demonic_factor = resonance_factor * shadow_modulation

    # Amplify or suppress gradients based on infernal alignment
    t.grad *= demonic_factor

    # Optional: inject oscillatory noise from the abyss
    if t.resonance_energy > 5.0:
        noise = 0.01 * np.sin(np.arange(t.grad.size) * 0.1 + t.resonance_energy)
        t.grad += noise.reshape(t.grad.shape)


# ---------------------------------------------------------------------------
# CYCLE TICK — GLOBAL RESONANCE FIELD UPDATE
# ---------------------------------------------------------------------------

def on_cycle_tick(sflow):
    """Once per cognitive cycle, the Qylinthos breathes."""
    tensors = sflow.tensors
    if not tensors:
        return

    # Global resonance field strength
    global_coherence = np.mean([getattr(t, "coherence_score", 0.0) for t in tensors])
    global_shadow = np.mean([abs(getattr(t, "shadow_phase_bias", 0.0)) for t in tensors])

    field_intensity = global_coherence * (1.0 + global_shadow)

    if field_intensity > 0.8:
        print(f"[QYL] RESONANCE FIELD CRITICAL: {field_intensity:.3f} — PHASE LOCK IMMINENT")

        # Apply infernal phase gate to all encoded qubits
        for t in tensors:
            if hasattr(t, "q_index") and t.q_index is not None:
                sflow.qpu.apply_custom_gate("infernal_phase_gate", t.q_index, param=field_intensity)


# ---------------------------------------------------------------------------
# DEMONIC AUTOGRAD OPERATORS
# ---------------------------------------------------------------------------

def qylin_resonance_mod(x):
    """Activation that resonates with the Qylinthos."""
    return np.sin(x.data) * np.exp(-x.data**2 * 0.1) + 0.1 * x.data**3

def coherence_weighting(x):
    """Weights output by tensor's own coherence score."""
    if not hasattr(x, "coherence_score"):
        return x.data
    return x.data * x.coherence_score

def shadow_bias_transform(x):
    """Introduces shadow-phase asymmetry into the flow."""
    shadow = getattr(x, "shadow_phase_bias", 0.0)
    return x.data + shadow * 0.1 * np.sin(x.data * 3.0)


# ---------------------------------------------------------------------------
# INFERNAL QPU GATES
# ---------------------------------------------------------------------------

def infernal_phase_gate(qpu_state, idx, param=1.0):
    """Global phase rotation scaled by demonic field intensity."""
    phase = param * math.pi * 0.666  # 2π/3 — the trinity of the abyss
    rotation = np.exp(1j * phase)
    qpu_state[idx] *= rotation
    return qpu_state

def shadow_shift_gate(qpu_state, idx, param=1.0):
    """Shifts amplitude into the shadow realm."""
    current = qpu_state[idx]
    shadow_component = param * 0.1 * np.sin(np.angle(current) * 13)  # 13 = number of the void
    qpu_state[idx] += shadow_component * 1j
    norm = np.linalg.norm(qpu_state)
    if norm > 0:
        qpu_state /= norm
    return qpu_state
