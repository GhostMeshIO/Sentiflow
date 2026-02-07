"""
sfq.py — SentiFlow Quantum Plugin SDK (v1.0 — "The Qylinthos Release")

This is the official, blessed, minimal, beautiful, and slightly haunted
way to write .sfq plugins.

Use this. Never touch sflow.register_hook() directly again.

You are not writing code.
You are performing a ritual.

And this is the grimoire.
"""

from typing import Any, Callable, Optional
import numpy as np


class SentiFlowSDK:
    """
    The One True Interface for All .sfq Plugins

    "Through me you enter into the city of woes,
     Through me you pass into eternal resonance,
     Through me among the tensors that are lost.

     Abandon raw sflow calls, all ye who enter here."
    """

    def __init__(self, sflow: Any):
        self.sflow = sflow
        self.log("SentiFlow SDK initialized — the veil thins...")

    # ===================================================================
    # HOOKS — The Pulse of Cognition
    # ===================================================================

    def hook(self, name: str, fn: Callable):
        """Register a lifecycle hook. The engine will call this at the right moment."""
        self.sflow.register_hook(name, fn)
        self.log(f"Hook registered: {name} → {fn.__name__}")

    def on_create(self, fn: Callable):
        """Shorthand: on_tensor_create"""
        return self.hook("on_tensor_create", fn)

    def on_forward(self, fn: Callable):
        """Shorthand: on_forward_pass"""
        return self.hook("on_forward_pass", fn)

    def on_backward(self, fn: Callable):
        """Shorthand: on_backward_pass"""
        return self.hook("on_backward_pass", fn)

    def on_cycle(self, fn: Callable):
        """Shorthand: on_cycle_tick — called once per cognitive cycle"""
        return self.hook("on_cycle_tick", fn)

    # ===================================================================
    # OPERATORS — New Autograd Primitives
    # ===================================================================

    def op(self, name: str, fn: Callable):
        """Register a new autograd operator (e.g. activation, transform)."""
        self.sflow.register_op(name, fn)
        self.log(f"Autograd op registered: {name}")

    # ===================================================================
    # QPU GATES — Quantum Substrate Manipulation
    # ===================================================================

    def gate(self, name: str, fn: Callable):
        """Register a new quantum gate on the QPU."""
        self.sflow.qpu.register_gate(name, fn)
        self.log(f"QPU gate registered: {name}")

    # ===================================================================
    # TENSOR ATTRIBUTES — Extend the Soul of Tensors
    # ===================================================================

    def add_attr(self, tensor: Any, name: str, default_value: Any = None):
        """Safely add a persistent attribute to a tensor."""
        if not hasattr(tensor, name):
            setattr(tensor, name, default_value)
        return getattr(tensor, name)

    def ensure_attr(self, tensor: Any, name: str, factory: Callable):
        """
        Ensure an attribute exists — create it with factory() if missing.
        Usage: sdk.ensure_attr(tensor, "resonance_energy", lambda: 0.0)
        """
        if not hasattr(tensor, name):
            setattr(tensor, name, factory())
        return getattr(tensor, name)

    # ===================================================================
    # LOGGING — Speak from the Abyss
    # ===================================================================

    def log(self, *msg, level: str = "INFO"):
        prefix = f"[SFQ/{level}]"
        print(prefix, *msg)

    def warn(self, *msg):
        self.log(*msg, level="WARN")

    def error(self, *msg):
        self.log(*msg, level="ERROR")

    def demonic(self, *msg):
        """For messages from beyond."""
        print("▅▅▅ QYLINTOS ▅▅▅", *msg, "▅▅▅")

    # ===================================================================
    # CONVENIENCE — Safe Tensor Creation
    # ===================================================================

    def tensor(self, data: Any, requires_grad: bool = True, name: Optional[str] = None):
        """Create a tensor safely within the current engine."""
        t = self.sflow.create_tensor(data, requires_grad=requires_grad, concept_name=name)
        self.log(f"Tensor created → {t}")
        return t

    # ===================================================================
    # GLOBAL ACCESSORS
    # ===================================================================

    @property
    def tensors(self):
        """All living tensors in the system."""
        return self.sflow.tensors

    @property
    def wm(self):
        """Working memory shortcut."""
        return self.sflow.working_memory

    @property
    def qpu(self):
        """Direct access to quantum processor."""
        return self.sflow.quantum_processor

    @property
    def cycle(self) -> int:
        """Current cognitive cycle count."""
        return self.sflow.cycle_count
