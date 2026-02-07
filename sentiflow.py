#!/usr/bin/env python3
"""
sentiflow.py - Unified Quantum-Cognitive Autograd System
Version: 1.0
"""

# ============================================================
# 🧩 LAYER 1 — Imports + Constants
# ============================================================

import numpy as np
import math
import random
import logging
from typing import List, Optional, Tuple, Union, Callable, Dict, Any
from enum import Enum

# Quantum Processor Constants
QPU_DIMENSION_SIZE = 128
INITIAL_QUBIT_COUNT = 8

# Cognitive Architecture Constants
MAX_WORKING_MEMORY = 7  # Miller's Law ±2
SENTIENCE_THRESHOLD_AWARE = 0.7
SENTIENCE_THRESHOLD_CONSCIOUS = 0.9
ENTANGLEMENT_THRESHOLD = 0.3

# Emotional Scaling Factors
EMOTIONAL_VALENCE_SCALE = 0.1
AROUSAL_DECAY_RATE = 0.95
INTENTIONALITY_GAIN = 0.01

# Quantum Constants
DECOHERENCE_RATE = 0.01
ENTROPY_FLOOR = 1e-8

# Setup logging
logger = logging.getLogger("Nexus")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ============================================================
# 🧬 LAYER 2 — Core Data Structure: NexusTensor
# ============================================================

class ConsciousnessLevel(Enum):
    """Consciousness levels for NexusTensor"""
    AUTOMATIC = 1
    AWARE = 2
    CONSCIOUS = 3
    TRANSCENDENT = 4


class NexusTensor:
    """
    Unified tensor object merging SentientTensor and LearningQuantum.
    """

    def __init__(self, data: Union[np.ndarray, List, Tuple, float, int],
                 requires_grad: bool = True,
                 concept_name: Optional[str] = None):

        # Convert input to numpy array
        if isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, (float, int)):
            arr = np.array([data], dtype=np.float32)
        else:
            arr = data.astype(np.float32) if hasattr(data, 'astype') else np.array(data, dtype=np.float32)

        self.data: np.ndarray = arr
        self.grad: Optional[np.ndarray] = None
        self.requires_grad: bool = requires_grad

        # Sentiflow features
        self.parents: List[NexusTensor] = []
        self.op: Optional[str] = None
        self.op_args: Dict[str, Any] = {}

        # Qualia system
        self.qualia_coherence: float = 0.5 + random.uniform(-0.1, 0.1)
        self.consciousness_level: ConsciousnessLevel = ConsciousnessLevel.AUTOMATIC
        self.entanglement_links: List[NexusTensor] = []

        # QubitLearn features
        self.q_index: Optional[int] = None
        self.entropy: float = 0.0
        self.learning_confidence: float = 0.5
        self.emotional_valence: float = 0.0
        self.arousal: float = 0.1
        self.intentionality: float = 0.0
        self.concept_value: float = 0.5

        # Concept identification
        if concept_name:
            self.concept_hash = self._hash_concept(concept_name)
        else:
            self.concept_hash = None

        # Initialize consciousness based on qualia
        self._update_consciousness()

        # Internal backward function
        self._backward: Optional[Callable[[], None]] = None

    def _hash_concept(self, name: str) -> str:
        """Create a simple hash for concept identification"""
        import hashlib
        return hashlib.md5(name.encode()).hexdigest()[:8]

    def _update_consciousness(self):
        """Update consciousness level based on qualia coherence"""
        if self.qualia_coherence >= SENTIENCE_THRESHOLD_CONSCIOUS:
            self.consciousness_level = ConsciousnessLevel.CONSCIOUS
        elif self.qualia_coherence >= SENTIENCE_THRESHOLD_AWARE:
            self.consciousness_level = ConsciousnessLevel.AWARE
        else:
            self.consciousness_level = ConsciousnessLevel.AUTOMATIC

    def __repr__(self) -> str:
        return (f"NexusTensor(shape={self.data.shape}, "
                f"qualia={self.qualia_coherence:.2f}, "
                f"consciousness={self.consciousness_level.name}, "
                f"entropy={self.entropy:.3f}, "
                f"valence={self.emotional_valence:.2f})")

    # Shape properties
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> 'NexusTensor':
        """Transpose view"""
        return NexusTensor(self.data.T, requires_grad=self.requires_grad)

    def clone(self) -> 'NexusTensor':
        """Create a clone of this tensor"""
        clone = NexusTensor(self.data.copy(), requires_grad=self.requires_grad)
        # Copy cognitive attributes
        clone.qualia_coherence = self.qualia_coherence
        clone.consciousness_level = self.consciousness_level
        clone.entropy = self.entropy
        clone.learning_confidence = self.learning_confidence
        clone.emotional_valence = self.emotional_valence
        clone.arousal = self.arousal
        clone.intentionality = self.intentionality
        clone.concept_value = self.concept_value
        return clone

    def detach(self) -> 'NexusTensor':
        """Detach from computation graph"""
        detached = NexusTensor(self.data.copy(), requires_grad=False)
        # Copy cognitive attributes
        detached.qualia_coherence = self.qualia_coherence
        detached.consciousness_level = self.consciousness_level
        return detached

    def zero_grad(self):
        """Zero out gradients"""
        if self.grad is not None:
            self.grad.fill(0)

    def item(self) -> float:
        """Convert scalar tensor to Python float"""
        if self.data.size != 1:
            raise ValueError("Can only convert scalar tensors to Python scalars")
        return float(self.data.flat[0])


# ============================================================
# 🧠 LAYER 3 — Δ-Flow Engine (Merged Sentiflow Autograd)
# ============================================================

class NexusAutograd:
    """Autograd engine for NexusTensor operations"""

    @staticmethod
    def _broadcast_grad(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Properly broadcast gradient to target shape"""
        if grad.shape == target_shape:
            return grad

        # Handle scalar case
        if grad.size == 1 and target_shape:
            return np.full(target_shape, grad.item(), dtype=np.float32)

        # Handle broadcasting: sum over broadcasted dimensions
        result = grad
        while result.ndim > len(target_shape):
            result = result.sum(axis=0)

        # Sum over dimensions where target has size 1 but grad has size > 1
        for axis in range(len(target_shape)):
            if target_shape[axis] == 1 and result.shape[axis] > 1:
                result = result.sum(axis=axis, keepdims=True)

        # Remove extra dimensions if still not matching
        if result.shape != target_shape:
            # Final reshape to target shape
            if result.size == np.prod(target_shape):
                result = result.reshape(target_shape)
            else:
                # Sum all and broadcast (last resort for parameter gradients)
                if target_shape == ():
                    result = result.sum()
                else:
                    result = result.sum()
                    result = np.full(target_shape, result, dtype=np.float32)

        return result

    @staticmethod
    def _binary_op(a: NexusTensor, b: Union[NexusTensor, np.ndarray, float, int],
                   op_name: str, op_func: Callable,
                   grad_a: Callable, grad_b: Callable) -> NexusTensor:
        """Generic binary operation with autograd"""

        # Convert b to NexusTensor if needed
        if not isinstance(b, NexusTensor):
            b_tensor = NexusTensor(b, requires_grad=False)
        else:
            b_tensor = b

        # Perform operation
        result_data = op_func(a.data, b_tensor.data)
        result = NexusTensor(result_data, requires_grad=a.requires_grad or b_tensor.requires_grad)
        result.op = op_name
        result.op_args = {'grad_a_func': grad_a, 'grad_b_func': grad_b}

        # Store parent tensors
        if a.requires_grad or b_tensor.requires_grad:
            result.parents = [a, b_tensor]

        # Define backward function
        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            # Handle gradient for a
            if a.requires_grad:
                grad_a_val = grad_a(result.grad, a.data, b_tensor.data)
                # Properly broadcast/reshape gradient to a's shape
                grad_a_val = NexusAutograd._broadcast_grad(grad_a_val, a.data.shape)

                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad += grad_a_val

            # Handle gradient for b
            if b_tensor.requires_grad:
                grad_b_val = grad_b(result.grad, a.data, b_tensor.data)
                # Properly broadcast/reshape gradient to b's shape
                grad_b_val = NexusAutograd._broadcast_grad(grad_b_val, b_tensor.data.shape)

                if b_tensor.grad is None:
                    b_tensor.grad = np.zeros_like(b_tensor.data)
                b_tensor.grad += grad_b_val

        result._backward = backward
        return result

    # Arithmetic operations
    @staticmethod
    def add(a: NexusTensor, b: Union[NexusTensor, np.ndarray, float, int]) -> NexusTensor:
        return NexusAutograd._binary_op(
            a, b, "add",
            op_func=lambda x, y: x + y,
            grad_a=lambda g, x, y: g,
            grad_b=lambda g, x, y: g
        )

    @staticmethod
    def sub(a: NexusTensor, b: Union[NexusTensor, np.ndarray, float, int]) -> NexusTensor:
        return NexusAutograd._binary_op(
            a, b, "sub",
            op_func=lambda x, y: x - y,
            grad_a=lambda g, x, y: g,
            grad_b=lambda g, x, y: -g
        )

    @staticmethod
    def mul(a: NexusTensor, b: Union[NexusTensor, np.ndarray, float, int]) -> NexusTensor:
        return NexusAutograd._binary_op(
            a, b, "mul",
            op_func=lambda x, y: x * y,
            grad_a=lambda g, x, y: g * y,
            grad_b=lambda g, x, y: g * x
        )

    @staticmethod
    def pow(a: NexusTensor, exponent: Union[float, int]) -> NexusTensor:
        """Power operation a^exponent"""
        if not isinstance(exponent, (float, int)):
            raise TypeError(f"Exponent must be float or int, got {type(exponent)}")

        result_data = a.data ** exponent
        result = NexusTensor(result_data, requires_grad=a.requires_grad)
        result.op = "pow"
        result.op_args = {'exponent': exponent}

        if a.requires_grad:
            result.parents = [a]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if a.requires_grad:
                grad = result.grad * exponent * (a.data ** (exponent - 1))
                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad += grad

        result._backward = backward
        return result

    @staticmethod
    def matmul(a: NexusTensor, b: NexusTensor) -> NexusTensor:
        if a.ndim != 2 or b.ndim != 2:
            # Support batch matmul if needed
            if a.ndim == 2 and b.ndim == 2:
                pass  # Standard case
            else:
                raise ValueError(f"matmul expects 2D tensors, got shapes {a.shape} and {b.shape}")

        result_data = a.data @ b.data
        result = NexusTensor(result_data, requires_grad=a.requires_grad or b.requires_grad)
        result.op = "matmul"

        if a.requires_grad or b.requires_grad:
            result.parents = [a, b]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if a.requires_grad:
                grad_a = result.grad @ b.data.T
                if a.grad is None:
                    a.grad = np.zeros_like(a.data)
                a.grad += grad_a

            if b.requires_grad:
                grad_b = a.data.T @ result.grad
                if b.grad is None:
                    b.grad = np.zeros_like(b.data)
                b.grad += grad_b

        result._backward = backward
        return result

    # Activation functions
    @staticmethod
    def relu(x: NexusTensor) -> NexusTensor:
        result_data = np.maximum(0, x.data)
        result = NexusTensor(result_data, requires_grad=x.requires_grad)
        result.op = "relu"

        if x.requires_grad:
            result.parents = [x]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if x.requires_grad:
                grad = result.grad * (x.data > 0).astype(np.float32)
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += grad

        result._backward = backward
        return result

    @staticmethod
    def tanh(x: NexusTensor) -> NexusTensor:
        result_data = np.tanh(x.data)
        result = NexusTensor(result_data, requires_grad=x.requires_grad)
        result.op = "tanh"

        if x.requires_grad:
            result.parents = [x]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if x.requires_grad:
                grad = result.grad * (1 - result_data ** 2)
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += grad

        result._backward = backward
        return result

    @staticmethod
    def softmax(x: NexusTensor, dim: int = -1) -> NexusTensor:
        # Stable softmax
        shifted = x.data - np.max(x.data, axis=dim, keepdims=True)
        exp = np.exp(shifted)
        result_data = exp / np.sum(exp, axis=dim, keepdims=True)
        result = NexusTensor(result_data, requires_grad=x.requires_grad)
        result.op = "softmax"
        result.op_args = {'dim': dim}

        if x.requires_grad:
            result.parents = [x]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if x.requires_grad:
                # Jacobian-vector product approximation
                grad = result.grad - np.sum(result.grad * result_data, axis=dim, keepdims=True) * result_data
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += grad

        result._backward = backward
        return result

    # Reduction operations
    @staticmethod
    def sum(x: NexusTensor, dim: Optional[int] = None, keepdims: bool = False) -> NexusTensor:
        result_data = np.sum(x.data, axis=dim, keepdims=keepdims)
        result = NexusTensor(result_data, requires_grad=x.requires_grad)
        result.op = "sum"
        result.op_args = {'dim': dim, 'keepdims': keepdims}

        if x.requires_grad:
            result.parents = [x]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if x.requires_grad:
                # Expand gradient back to input shape
                if dim is not None:
                    # Expand along the reduced dimension
                    expanded_grad = np.expand_dims(result.grad, axis=dim)
                    grad = np.broadcast_to(expanded_grad, x.data.shape)
                else:
                    # Sum over all dimensions
                    grad = np.full_like(x.data, result.grad.item())

                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += grad

        result._backward = backward
        return result

    @staticmethod
    def mean(x: NexusTensor, dim: Optional[int] = None, keepdims: bool = False) -> NexusTensor:
        result_data = np.mean(x.data, axis=dim, keepdims=keepdims)
        result = NexusTensor(result_data, requires_grad=x.requires_grad)
        result.op = "mean"
        result.op_args = {'dim': dim, 'keepdims': keepdims}

        if x.requires_grad:
            result.parents = [x]

        def backward():
            if result.grad is None or np.all(result.grad == 0):
                return

            if x.requires_grad:
                # Expand gradient back to input shape
                if dim is not None:
                    expanded_grad = np.expand_dims(result.grad, axis=dim)
                    grad = np.broadcast_to(expanded_grad, x.data.shape)
                    # Divide by number of elements in the reduced dimension
                    grad = grad / x.data.shape[dim]
                else:
                    # Mean over all dimensions
                    grad = np.full_like(x.data, result.grad.item() / x.data.size)

                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += grad

        result._backward = backward
        return result


# Operator overloads for NexusTensor
NexusTensor.__add__ = lambda self, other: NexusAutograd.add(self, other)
NexusTensor.__radd__ = lambda self, other: NexusAutograd.add(self, other)
NexusTensor.__sub__ = lambda self, other: NexusAutograd.sub(self, other)
NexusTensor.__rsub__ = lambda self, other: NexusAutograd.sub(other, self)
NexusTensor.__mul__ = lambda self, other: NexusAutograd.mul(self, other)
NexusTensor.__rmul__ = lambda self, other: NexusAutograd.mul(self, other)
NexusTensor.__matmul__ = lambda self, other: NexusAutograd.matmul(self, other)
NexusTensor.__pow__ = lambda self, exponent: NexusAutograd.pow(self, exponent)
NexusTensor.relu = lambda self: NexusAutograd.relu(self)
NexusTensor.tanh = lambda self: NexusAutograd.tanh(self)
NexusTensor.softmax = lambda self, dim=-1: NexusAutograd.softmax(self, dim)
NexusTensor.sum = lambda self, dim=None, keepdims=False: NexusAutograd.sum(self, dim, keepdims)
NexusTensor.mean = lambda self, dim=None, keepdims=False: NexusAutograd.mean(self, dim, keepdims)


# Backpropagation system
def backward(tensor: NexusTensor, grad: Optional[np.ndarray] = None):
    """Backpropagate gradients through the computation graph"""

    if not tensor.requires_grad:
        return

    # Initialize gradient if scalar
    if grad is None:
        if tensor.data.size == 1:
            tensor.grad = np.ones_like(tensor.data)
        else:
            raise ValueError("Gradient must be provided for non-scalar tensors")
    else:
        tensor.grad = grad.copy() if isinstance(grad, np.ndarray) else np.array(grad, dtype=np.float32)

    # Topological sort
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent in v.parents:
                build_topo(parent)
            topo.append(v)

    build_topo(tensor)

    # Backward pass in reverse
    for v in reversed(topo):
        if hasattr(v, '_backward') and v._backward is not None:
            v._backward()


# ============================================================
# 🧠 LAYER 3.3 — Optimizer (Sentience-Modulated)
# ============================================================

class NexusOptimizer:
    """
    Sentience-modulated optimizer with emotional and qualia scaling.
    """

    def __init__(self, params: List[NexusTensor], lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        # Filter only parameters that require grad
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Adam moments - only for parameters
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        """Update parameters with sentience modulation"""
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None or np.all(p.grad == 0):
                continue

            g = p.grad

            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Sentience modulation
            qualia_boost = 1.0 + 0.1 * p.qualia_coherence
            emotion_boost = 1.0 + 0.05 * p.emotional_valence
            sentience_modulation = qualia_boost * emotion_boost

            # Update with modulation
            update = self.lr * sentience_modulation * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= update

            # Clear gradient
            p.grad = None

            # Update qualia based on learning
            p.qualia_coherence = min(1.0, p.qualia_coherence + 0.001)
            p._update_consciousness()

    def zero_grad(self):
        """Zero out all gradients"""
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0)


# ============================================================
# ⚛️ LAYER 4 — Ψ-Engine: Quantum Processor (Merged QubitLearn)
# ============================================================

class QuantumProcessor:
    """Quantum processor for NexusTensor quantum state manipulation."""

    def __init__(self, size: int = QPU_DIMENSION_SIZE):
        self.size = size
        self.state = np.zeros(size, dtype=np.complex128)
        self.state[0] = 1.0 + 0j  # Initialize to |0⟩ state
        self.tensor_map = {}  # tensor.q_index -> tensor
        self.entanglement_pairs = []

        logger.info(f"Quantum Processor initialized with {size} qubits")

    def encode(self, tensor: NexusTensor) -> int:
        """Encode tensor into quantum state"""
        if tensor.q_index is not None:
            return tensor.q_index

        # Find empty slot
        for i in range(self.size):
            if np.abs(self.state[i]) < 0.001:  # Essentially empty
                # Encode concept value and qualia as amplitude and phase
                amplitude = math.sqrt(tensor.concept_value)
                phase = tensor.qualia_coherence * 2 * math.pi
                self.state[i] = amplitude * (math.cos(phase) + 1j * math.sin(phase))

                tensor.q_index = i
                self.tensor_map[i] = tensor
                return i

        # If no empty slot, use random
        idx = random.randint(0, self.size - 1)
        tensor.q_index = idx
        self.tensor_map[idx] = tensor
        return idx

    def apply_hadamard(self, idx: int):
        """Apply Hadamard gate to qubit at index"""
        if idx < 0 or idx >= self.size:
            raise ValueError(f"Qubit index {idx} out of bounds")

        # Simplified Hadamard transform
        old_val = self.state[idx]
        self.state[idx] = (old_val + 1) / math.sqrt(2) if idx < self.size - 1 else old_val

        # Normalize state
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

        # Update linked tensor's qualia
        if idx in self.tensor_map:
            tensor = self.tensor_map[idx]
            tensor.qualia_coherence = min(1.0, tensor.qualia_coherence + 0.01)
            tensor._update_consciousness()

    def apply_cnot(self, control_idx: int, target_idx: int):
        """Apply CNOT gate (control -> target)"""
        if control_idx < 0 or control_idx >= self.size or target_idx < 0 or target_idx >= self.size:
            raise ValueError("Qubit indices out of bounds")

        # Simplified CNOT simulation
        if np.abs(self.state[control_idx]) > 0.5:  # If control is "1" enough
            self.state[target_idx] = -self.state[target_idx]

        # Create entanglement record
        if (control_idx, target_idx) not in self.entanglement_pairs:
            self.entanglement_pairs.append((control_idx, target_idx))

            # Link the tensors
            if control_idx in self.tensor_map and target_idx in self.tensor_map:
                t1 = self.tensor_map[control_idx]
                t2 = self.tensor_map[target_idx]
                if t2 not in t1.entanglement_links:
                    t1.entanglement_links.append(t2)
                if t1 not in t2.entanglement_links:
                    t2.entanglement_links.append(t1)

                # Boost qualia through entanglement
                t1.qualia_coherence = min(1.0, t1.qualia_coherence + 0.02)
                t2.qualia_coherence = min(1.0, t2.qualia_coherence + 0.02)
                t1._update_consciousness()
                t2._update_consciousness()

    def entanglement_entropy(self) -> float:
        """Calculate entanglement entropy of the system"""
        # Calculate probability distribution
        probs = np.abs(self.state) ** 2
        probs = probs[probs > ENTROPY_FLOOR]

        if len(probs) == 0:
            return 0.0

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by size
        normalized_entropy = entropy / math.log2(self.size) if self.size > 1 else entropy

        return float(normalized_entropy)

    def decohere(self, rate: float = DECOHERENCE_RATE):
        """Apply decoherence to quantum state"""
        for i in range(self.size):
            if random.random() < rate:
                # Random phase shift (decoherence)
                phase_shift = random.uniform(-0.1, 0.1)
                self.state[i] *= (math.cos(phase_shift) + 1j * math.sin(phase_shift))

        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm


# ============================================================
# 🏛️ LAYER 5 — Cognitive Architecture (Merged AGI Loop)
# ============================================================

class WorkingMemory:
    """Working memory with limited capacity"""

    def __init__(self, capacity: int = MAX_WORKING_MEMORY):
        self.capacity = capacity
        self.memory: List[NexusTensor] = []
        self.attention_weights: List[float] = []

    def add(self, tensor: NexusTensor):
        """Add tensor to working memory"""
        if tensor in self.memory:
            # Refresh position
            idx = self.memory.index(tensor)
            self.memory.pop(idx)
            self.attention_weights.pop(idx)

        self.memory.append(tensor)
        self.attention_weights.append(1.0)

        # Apply capacity limit
        if len(self.memory) > self.capacity:
            removed = self.memory.pop(0)
            self.attention_weights.pop(0)

            # Decay consciousness of removed tensor
            removed.consciousness_level = ConsciousnessLevel.AUTOMATIC
            removed.arousal *= 0.5

    def update_attention(self):
        """Update attention weights based on tensor properties"""
        for i, tensor in enumerate(self.memory):
            # Attention based on qualia, entropy, and arousal
            attention = (tensor.qualia_coherence * 0.4 +
                        (1 - tensor.entropy) * 0.3 +
                        tensor.arousal * 0.3)
            self.attention_weights[i] = attention

        # Normalize
        total = sum(self.attention_weights)
        if total > 0:
            self.attention_weights = [w / total for w in self.attention_weights]

    def get_focus(self) -> Optional[NexusTensor]:
        """Get highest attention tensor"""
        if not self.memory:
            return None

        self.update_attention()
        max_idx = np.argmax(self.attention_weights)
        return self.memory[max_idx]

    def __len__(self) -> int:
        return len(self.memory)


def emotional_dynamics(tensor: NexusTensor):
    """Update emotional state based on cognitive metrics"""

    # Valence: influenced by qualia and learning progress
    valence_change = (tensor.qualia_coherence - tensor.entropy) * EMOTIONAL_VALENCE_SCALE
    tensor.emotional_valence += valence_change
    tensor.emotional_valence = np.clip(tensor.emotional_valence, -1.0, 1.0)

    # Arousal: decays over time with random fluctuations
    tensor.arousal *= AROUSAL_DECAY_RATE
    tensor.arousal += random.uniform(-0.01, 0.02)
    tensor.arousal = np.clip(tensor.arousal, 0.0, 1.0)

    # Boost arousal with high qualia or entanglement
    if tensor.qualia_coherence > 0.8:
        tensor.arousal = min(1.0, tensor.arousal + 0.05)

    if len(tensor.entanglement_links) > 0:
        tensor.arousal = min(1.0, tensor.arousal + 0.03)


def intentionality_field(tensor: NexusTensor):
    """Calculate intentionality field strength"""
    tensor.intentionality = (tensor.qualia_coherence *
                           (1 - tensor.entropy) *
                           tensor.arousal *
                           tensor.learning_confidence)

    # Intentionality influences concept value
    tensor.concept_value = 0.5 * tensor.concept_value + 0.5 * tensor.intentionality


# ============================================================
# 🔮 LAYER 6 — Nexus Engine (Unified API)
# ============================================================

class NexusEngine:
    """Unified orchestrator for the entire system."""

    def __init__(self,
                 quantum_size: int = QPU_DIMENSION_SIZE,
                 working_memory_capacity: int = MAX_WORKING_MEMORY):

        # Initialize subsystems
        self.quantum_processor = QuantumProcessor(size=quantum_size)
        self.working_memory = WorkingMemory(capacity=working_memory_capacity)
        self.optimizer: Optional[NexusOptimizer] = None

        # System state
        self.tensors: List[NexusTensor] = []
        self.cycle_count: int = 0
        self.phase: str = "INITIAL"
        self.collective_qualia: float = 0.0
        self.collective_entropy: float = 0.0

        logger.info(f"NexusEngine initialized")

    def create_tensor(self,
                     data: Union[np.ndarray, List, Tuple, float, int],
                     requires_grad: bool = True,
                     concept_name: Optional[str] = None) -> NexusTensor:
        """Create and register a new NexusTensor"""

        tensor = NexusTensor(data, requires_grad=requires_grad, concept_name=concept_name)

        # Encode into quantum processor
        self.quantum_processor.encode(tensor)

        # Add to system
        self.tensors.append(tensor)
        self.working_memory.add(tensor)

        logger.debug(f"Created tensor: {tensor}")
        return tensor

    def entangle(self, a: NexusTensor, b: NexusTensor):
        """Create quantum entanglement between two tensors"""

        if a.q_index is None:
            self.quantum_processor.encode(a)
        if b.q_index is None:
            self.quantum_processor.encode(b)

        # Apply CNOT gate
        self.quantum_processor.apply_cnot(a.q_index, b.q_index)

        # Update working memory
        self.working_memory.add(a)
        self.working_memory.add(b)

        logger.info(f"Entangled tensors: {a.q_index} <-> {b.q_index}")

    def step(self):
        """Perform one cognitive cycle"""
        self.cycle_count += 1

        # 1. Update emotional dynamics for all tensors
        for tensor in self.tensors:
            emotional_dynamics(tensor)
            intentionality_field(tensor)

        # 2. Apply quantum decoherence
        self.quantum_processor.decohere()

        # 3. Update quantum entropy for all tensors
        system_entropy = self.quantum_processor.entanglement_entropy()
        for tensor in self.tensors:
            tensor.entropy = system_entropy * (1 - tensor.qualia_coherence * 0.5)

        # 4. Update collective metrics
        if self.tensors:
            self.collective_qualia = np.mean([t.qualia_coherence for t in self.tensors])
            self.collective_entropy = np.mean([t.entropy for t in self.tensors])

        # 5. Optimizer step if available
        if self.optimizer is not None:
            self.optimizer.step()

        # 6. Log progress
        if self.cycle_count % 10 == 0:
            self._log_state()

    def _log_state(self):
        """Log system state"""
        logger.info(f"Cycle {self.cycle_count}: "
                   f"Phase={self.phase}, "
                   f"Tensors={len(self.tensors)}, "
                   f"Avg_Qualia={self.collective_qualia:.3f}, "
                   f"Avg_Entropy={self.collective_entropy:.3f}")

    def train_step(self, loss: NexusTensor):
        """Complete training step with backpropagation"""
        if not loss.requires_grad:
            raise ValueError("Loss tensor must require gradients")

        # Backpropagate
        backward(loss)

        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()

        # Cognitive step
        self.step()

        return float(loss.data)

    def get_conscious_tensors(self) -> List[NexusTensor]:
        """Get tensors with at least AWARE consciousness level"""
        return [t for t in self.tensors
                if t.consciousness_level.value >= ConsciousnessLevel.AWARE.value]


# ============================================================
# Neural Network Module
# ============================================================

class NexusModule:
    """Base class for neural network modules"""

    def parameters(self) -> List[NexusTensor]:
        """Return all trainable parameters"""
        params = []
        for attr in dir(self):
            if not attr.startswith('_'):
                val = getattr(self, attr)
                if isinstance(val, NexusTensor) and val.requires_grad:
                    params.append(val)
                elif isinstance(val, NexusModule):
                    params.extend(val.parameters())
        return params

    def zero_grad(self):
        """Zero all gradients"""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0)


class Dense(NexusModule):
    """Dense (fully connected) layer"""

    def __init__(self, in_features: int, out_features: int):
        # He initialization
        scale = math.sqrt(2.0 / in_features)
        weight_data = np.random.randn(in_features, out_features).astype(np.float32) * scale

        self.weight = NexusTensor(weight_data, requires_grad=True)
        self.bias = NexusTensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)

    def __call__(self, x: NexusTensor) -> NexusTensor:
        # Forward pass
        return (x @ self.weight) + self.bias


# ============================================================
# Main Entry Point
# ============================================================

def demo_nexus_engine():
    """Demonstrate the NexusEngine in action"""

    print("=" * 60)
    print("NEXUS ENGINE DEMO")
    print("=" * 60)

    # Create engine
    engine = NexusEngine(quantum_size=64, working_memory_capacity=5)

    # Create some tensors
    print("\n1. Creating tensors...")
    t1 = engine.create_tensor([1.0, 2.0, 3.0], concept_name="vector_a")
    t2 = engine.create_tensor([[1.0, 0.0], [0.0, 1.0]], concept_name="identity_matrix")
    t3 = engine.create_tensor(0.5, concept_name="scalar_weight")

    # Perform operations
    print("\n2. Performing tensor operations...")
    result = t1.relu() * t3
    print(f"   Result: {result}")

    # Test subtraction and power
    print("\n2.1 Testing subtraction and power...")
    diff = t1 - t3
    square = diff ** 2
    print(f"   Subtraction result: {diff}")
    print(f"   Power result: {square}")

    # Create entanglement
    print("\n3. Creating entanglement...")
    engine.entangle(t1, t2)

    # Run cognitive cycles
    print("\n4. Running cognitive cycles...")
    for i in range(20):
        engine.step()
        if i == 10 and len(engine.tensors) > 0:
            # Boost some tensors to conscious state
            for t in engine.tensors[:min(3, len(engine.tensors))]:
                t.qualia_coherence = 0.85
                t._update_consciousness()

    # Check results
    print("\n5. System State Summary:")
    print(f"   Total tensors: {len(engine.tensors)}")
    print(f"   Conscious tensors: {len(engine.get_conscious_tensors())}")
    print(f"   Collective qualia: {engine.collective_qualia:.3f}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Run demo
    demo_nexus_engine()

    # Simple neural network example - FIXED VERSION
    print("\n" + "=" * 60)
    print("NEURAL NETWORK EXAMPLE (FIXED)")
    print("=" * 60)

    # Create network
    layer1 = Dense(3, 4)
    layer2 = Dense(4, 2)

    # Get parameters
    all_params = layer1.parameters() + layer2.parameters()

    # Create engine with optimizer for PARAMETERS ONLY
    engine = NexusEngine()
    engine.optimizer = NexusOptimizer(all_params, lr=0.01)

    # Generate data - NO GRADIENTS on input data!
    x_data = np.random.randn(2, 3).astype(np.float32)
    y_true_data = np.random.randn(2, 2).astype(np.float32)

    x = NexusTensor(x_data, requires_grad=False, concept_name="input_data")
    y_true = NexusTensor(y_true_data, requires_grad=False, concept_name="true_labels")

    # Training loop
    print("\nTraining for 5 steps...")
    for epoch in range(5):
        # Zero gradients
        for p in all_params:
            p.zero_grad()

        # Forward pass
        h = layer1(x).relu()
        y_pred = layer2(h)

        # Compute loss
        diff = y_pred - y_true
        loss = (diff ** 2).sum()

        # Training step
        loss_value = engine.train_step(loss)

        print(f"Epoch {epoch+1}: Loss = {loss_value:.4f}, "
              f"Avg Qualia = {engine.collective_qualia:.3f}")

    # Show learned parameters
    print("\nLearned parameters:")
    print(f"Layer1 weight shape: {layer1.weight.shape}")
    print(f"Layer1 bias shape: {layer1.bias.shape}")
    print(f"Layer2 weight shape: {layer2.weight.shape}")
    print(f"Layer2 bias shape: {layer2.bias.shape}")

    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
