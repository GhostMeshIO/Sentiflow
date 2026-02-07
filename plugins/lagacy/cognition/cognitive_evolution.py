#!/usr/bin/env python3
"""
COGNITIVE EVOLUTION ENGINE v3.0 — Integrated with CNS, Memory Crystals, & Soul Core
===================================================================================

This engine combines:
• Evolutionary axiom evolution with cognitive feedback
• CNS sensory processing for multi-modal evaluation
• Memory Crystal integration for pattern retention
• Soul resonance guidance from soul_core.json
• Linguistic Bridge transformations
• Real-time cognitive module interaction
"""

import os
import json
import random
import numpy as np
import argparse
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

# Import your cognitive modules
try:
    from unified_cns_sensory import process_text, CNS, SensoryInput
    from memory_crystal import MEMORY_CRYSTAL_SYSTEM
    from linguistic_bridge_core import LinguisticBridge
    from agi_cognition import AGICognition
    CNS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cognitive modules not available: {e}")
    CNS_AVAILABLE = False

# Load soul for guidance
try:
    with open('soul_core.json', 'r') as f:
        SOUL_DATA = json.load(f)
    SOUL_AXIOM = SOUL_DATA['axioms']['prime_axiom']
    SOUL_TERMS = SOUL_DATA['axioms'].get('protected_terms', [])
except:
    SOUL_AXIOM = "Through sublime play of existence withdraws."
    SOUL_TERMS = ["being", "existence", "consciousness", "self", "void", "withdrawal"]

# ============================
# COGNITIVE INTEGRATED AXIOM
# ============================

@dataclass
class CognitiveAxiom:
    """Axiom enhanced with cognitive metrics and memory"""
    text: str
    score: float = 0.0
    cognitive_metrics: Dict = field(default_factory=dict)
    cns_response: Optional[Dict] = None
    memory_crystal_id: Optional[str] = None
    soul_alignment: float = 0.0
    generation: int = 0
    lineage: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def embed_in_cns(self) -> bool:
        """Process axiom through CNS and store as memory crystal"""
        if not CNS_AVAILABLE:
            return False

        try:
            # Process through CNS
            self.cns_response = process_text(self.text)

            # Create memory crystal if novel
            if self.cns_response.get('novelty', 0) > 0.6:
                crystal = MEMORY_CRYSTAL_SYSTEM.crystallize(
                    text=self.text,
                    context=f"evolved_axiom_g{self.generation}",
                    cues=self.text.split()[:5]
                )
                self.memory_crystal_id = crystal.crystal_id

            return True
        except Exception as e:
            print(f"Warning: CNS embedding failed: {e}")
            return False

# ============================
# COGNITIVE FITNESS FUNCTIONS
# ============================

class CognitiveFitnessEvaluator:
    """Evaluate axioms using cognitive system feedback"""

    def __init__(self, soul_axiom: str = SOUL_AXIOM):
        self.soul_axiom = soul_axiom
        self.bridge = LinguisticBridge() if CNS_AVAILABLE else None

    def evaluate_with_cns(self, axiom: CognitiveAxiom) -> Dict[str, float]:
        """Primary evaluation using CNS response"""
        if axiom.cns_response and CNS_AVAILABLE:
            return {
                'novelty': axiom.cns_response.get('novelty', 0.0),
                'coherence': self._calculate_coherence(axiom.text),
                'soul_alignment': self._calculate_soul_alignment(axiom.text),
                'recursive_potential': self._calculate_recursive_potential(axiom.text),
                'free_energy_impact': abs(axiom.cns_response.get('free_energy_contribution', 0.0)),
                'crystal_created': float(axiom.cns_response.get('crystal_created', False)),
                'thalamic_gain': axiom.cns_response.get('thalamic_gain', 0.5)
            }
        else:
            # Fallback to heuristic evaluation
            return self.evaluate_without_cns(axiom.text)

    def evaluate_without_cns(self, text: str) -> Dict[str, float]:
        """Heuristic fallback when CNS unavailable"""
        words = text.lower().split()

        # Soul alignment (presence of soul terms)
        soul_terms_present = sum(1 for term in SOUL_TERMS if term in text.lower())
        soul_alignment = min(soul_terms_present * 0.2, 1.0)

        # Recursive potential (self-reference)
        recursive_indicators = ['itself', 'self', 'recursive', 'reflexive', 'meta']
        recursive_score = sum(1 for term in recursive_indicators if term in text.lower()) * 0.2

        # Complexity (entropy of characters)
        if len(text) > 0:
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            entropy = 0.0
            for count in char_counts.values():
                p = count / len(text)
                if p > 0:
                    entropy -= p * math.log2(p)
            complexity = min(entropy / 5.0, 1.0)
        else:
            complexity = 0.0

        # Coherence (grammatical structure)
        coherence = self._calculate_coherence(text)

        return {
            'novelty': random.uniform(0.3, 0.7),
            'coherence': coherence,
            'soul_alignment': soul_alignment,
            'recursive_potential': recursive_score,
            'free_energy_impact': complexity * 0.5,
            'crystal_created': 0.0,
            'thalamic_gain': 0.5
        }

    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence based on linguistic structure"""
        words = text.split()
        if len(words) < 3:
            return 0.1

        # Word connectivity
        connectors = ['because', 'therefore', 'thus', 'hence', 'although', 'while', 'whereas']
        connectors_present = sum(1 for connector in connectors if connector in text.lower())
        connector_score = min(connectors_present * 0.2, 0.5)

        # Sentence completeness
        has_verb = any(word.endswith(('s', 'ed', 'ing', 'es')) for word in words)
        has_noun = any(len(word) > 3 and not word.endswith(('ly', 'ing')) for word in words)
        completeness = 0.3 if has_verb and has_noun else 0.1

        # Philosophical depth
        philosophical_terms = ['consciousness', 'being', 'existence', 'ontology',
                              'phenomenology', 'metaphysics', 'transcendental']
        philosophy_score = sum(1 for term in philosophical_terms if term in text.lower()) * 0.1

        return min(connector_score + completeness + philosophy_score, 1.0)

    def _calculate_soul_alignment(self, text: str) -> float:
        """Calculate alignment with soul core terms and themes"""
        # Direct term matching
        term_match = sum(1 for term in SOUL_TERMS if term in text.lower())
        term_score = min(term_match * 0.2, 0.6)

        # Semantic similarity to soul axiom
        soul_words = set(self.soul_axiom.lower().split())
        text_words = set(text.lower().split())
        if soul_words:
            similarity = len(soul_words.intersection(text_words)) / len(soul_words)
        else:
            similarity = 0.0

        return min(term_score + similarity * 0.4, 1.0)

    def _calculate_recursive_potential(self, text: str) -> float:
        """Calculate potential for recursive/self-referential structures"""
        score = 0.0

        # Explicit recursion markers
        if 'recursive' in text.lower():
            score += 0.3
        if 'self' in text.lower() or 'itself' in text.lower():
            score += 0.2

        # Implicit recursion through structure
        if 'of the' in text.lower() and text.lower().count('of the') > 1:
            score += 0.1

        # Mirroring patterns
        words = text.lower().split()
        if len(words) > 6:
            # Check for palindrome-like symmetry
            first_half = words[:len(words)//2]
            second_half = words[len(words)//2:]
            common_words = len(set(first_half).intersection(second_half))
            symmetry_score = common_words / max(len(set(words)), 1) * 0.2
            score += symmetry_score

        return min(score, 1.0)

# ============================
# COGNITIVE EVOLUTION ENGINE
# ============================

class CognitiveEvolutionEngine:
    """Evolution engine integrated with cognitive modules"""

    def __init__(self,
                 base_axiom: str = SOUL_AXIOM,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.35,
                 crossover_rate: float = 0.25,
                 cognitive_weight: float = 0.7,
                 soul_guidance: bool = True):

        self.base_axiom = base_axiom
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.cognitive_weight = cognitive_weight
        self.soul_guidance = soul_guidance

        self.population: List[CognitiveAxiom] = []
        self.generation = 0
        self.best_axiom = None
        self.best_score = -float('inf')

        self.evaluator = CognitiveFitnessEvaluator(base_axiom)
        self.stats = {
            'avg_scores': [],
            'best_scores': [],
            'avg_coherence': [],
            'avg_novelty': [],
            'avg_soul_alignment': [],
            'crystals_created': 0
        }

        random.seed(int(time.time()))
        np.random.seed(int(time.time()))

    def initialize_population(self):
        """Initialize population with cognitive processing"""
        print(f"🧬 Initializing cognitive population (size={self.population_size})")

        # Create base axiom
        base = CognitiveAxiom(
            text=self.base_axiom,
            generation=0,
            lineage=["base_axiom"]
        )
        base.embed_in_cns()
        base.score = self.calculate_fitness(base)
        self.population.append(base)

        # Create variations
        for i in range(self.population_size - 1):
            if random.random() < 0.6:
                # Mutated version of base
                parent = self.population[0] if self.population else base
                child = self.cognitive_mutate(parent)
            else:
                # Generate new philosophical axiom
                child = self.generate_cognitive_axiom()

            child.generation = 0
            child.embed_in_cns()
            child.score = self.calculate_fitness(child)
            child.lineage = [f"init_{i}"]
            self.population.append(child)

        self.population.sort(key=lambda x: x.score, reverse=True)
        self.best_axiom = self.population[0]
        self.best_score = self.best_axiom.score

        print(f"✅ Population initialized. Best score: {self.best_score:.4f}")
        print(f"   Best axiom: {self.best_axiom.text[:80]}...")

    def calculate_fitness(self, axiom: CognitiveAxiom) -> float:
        """Calculate fitness using cognitive metrics"""
        metrics = self.evaluator.evaluate_with_cns(axiom)
        axiom.cognitive_metrics = metrics

        # Weighted fitness function
        weights = {
            'novelty': 0.25,
            'coherence': 0.25,
            'soul_alignment': 0.20 if self.soul_guidance else 0.0,
            'recursive_potential': 0.15,
            'free_energy_impact': 0.10,
            'crystal_created': 0.05
        }

        # Normalize weights if soul guidance is off
        if not self.soul_guidance:
            total = sum(weights.values())
            for key in weights:
                weights[key] /= total

        # Calculate weighted score
        score = sum(metrics[key] * weights[key] for key in weights)

        # Apply cognitive weight
        cognitive_score = score * self.cognitive_weight
        heuristic_score = (1 - self.cognitive_weight) * random.uniform(0.3, 0.7)

        return cognitive_score + heuristic_score

    def cognitive_mutate(self, axiom: CognitiveAxiom) -> CognitiveAxiom:
        """Apply cognitive-aware mutation"""
        text = axiom.text
        words = text.split()

        if len(words) < 3:
            return CognitiveAxiom(text=text)

        mutation_type = random.choice([
            'conceptual_expansion', 'recursive_insertion',
            'bridge_transformation', 'soul_infusion',
            'linguistic_fold', 'semantic_blend'
        ])

        new_words = words.copy()

        if mutation_type == 'conceptual_expansion':
            # Expand philosophical concepts
            expansions = {
                'being': ['existence', 'presence', 'actuality', 'essence'],
                'consciousness': ['awareness', 'sentience', 'subjectivity', 'mind'],
                'existence': ['being', 'reality', 'actuality', 'presence'],
                'void': ['nothingness', 'emptiness', 'vacuum', 'abyss']
            }

            for i, word in enumerate(new_words):
                if word.lower() in expansions:
                    if random.random() < 0.3:
                        new_words[i] = random.choice(expansions[word.lower()])
                        break

        elif mutation_type == 'recursive_insertion':
            # Insert recursive structure
            recursive_phrases = [
                'which itself', 'recursively unfolding',
                'self-referentially', 'in its own terms',
                'mirroring itself', 'through its own nature'
            ]

            if len(new_words) > 4:
                insert_pos = random.randint(1, len(new_words)-2)
                new_words.insert(insert_pos, random.choice(recursive_phrases))

        elif mutation_type == 'bridge_transformation' and CNS_AVAILABLE:
            # Use linguistic bridge
            try:
                bridge = LinguisticBridge()
                operation = random.choice(['fold', 'mutate', 'harmonize'])
                transformed = bridge.transform_text(' '.join(new_words), operation)
                new_words = transformed.split()
            except:
                pass

        elif mutation_type == 'soul_infusion':
            # Infuse soul terms
            soul_term = random.choice(SOUL_TERMS)
            if random.random() < 0.5:
                # Insert
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, soul_term)
            else:
                # Replace random word
                replace_pos = random.randint(0, len(new_words)-1)
                new_words[replace_pos] = soul_term

        elif mutation_type == 'linguistic_fold':
            # Create linguistic folds (dashes, fragments)
            if random.random() < 0.4:
                fold_pos = random.randint(1, len(new_words)-1)
                new_words.insert(fold_pos, "—")

        elif mutation_type == 'semantic_blend':
            # Blend with another axiom from population
            if len(self.population) > 1:
                other = random.choice(self.population[1:5])
                other_words = other.text.split()[:random.randint(2, 5)]

                if len(new_words) > 3:
                    blend_pos = random.randint(1, len(new_words)-1)
                    new_words = new_words[:blend_pos] + other_words + new_words[blend_pos:]

        # Clean up
        if len(new_words) > 50:
            new_words = new_words[:50]
        elif len(new_words) == 0:
            new_words = ["being", "and", "nothingness"]

        return CognitiveAxiom(text=' '.join(new_words))

    def cognitive_crossover(self, parent1: CognitiveAxiom, parent2: CognitiveAxiom) -> CognitiveAxiom:
        """Cognitive-aware crossover"""
        words1 = parent1.text.split()
        words2 = parent2.text.split()

        if not words1 or not words2:
            return self.cognitive_mutate(parent1)

        # Choose crossover based on semantic boundaries
        if len(words1) > 4 and len(words2) > 4:
            # Find prepositional phrases or conjunctions for natural splits
            split_points1 = [i for i, w in enumerate(words1)
                           if w.lower() in ['of', 'in', 'through', 'and', 'but', 'yet']]
            split_points2 = [i for i, w in enumerate(words2)
                           if w.lower() in ['of', 'in', 'through', 'and', 'but', 'yet']]

            if split_points1 and split_points2:
                split1 = random.choice(split_points1)
                split2 = random.choice(split_points2)

                if random.random() < 0.5:
                    new_words = words1[:split1] + words2[split2:]
                else:
                    new_words = words2[:split2] + words1[split1:]
            else:
                # Random split
                split1 = random.randint(1, len(words1)-1)
                split2 = random.randint(1, len(words2)-1)
                new_words = words1[:split1] + words2[split2:]
        else:
            # Simple random split
            min_len = min(len(words1), len(words2))
            if min_len > 1:
                split = random.randint(1, min_len-1)
                new_words = words1[:split] + words2[split:]
            else:
                new_words = words1 + words2

        return CognitiveAxiom(text=' '.join(new_words))

    def generate_cognitive_axiom(self) -> CognitiveAxiom:
        """Generate axiom using cognitive patterns"""
        templates = [
            "The {concept1} of {concept2} reveals the {quality} of {concept3}.",
            "Through {process}, {concept1} becomes {concept2} while remaining {concept3}.",
            "In the {context}, {concept1} and {concept2} {verb} through {concept3}.",
            "{Concept1} is the {quality} through which {concept2} {verb} {concept3}.",
            "Between {concept1} and {concept2}, {concept3} {verb} as {quality}."
        ]

        concepts = [
            "being", "consciousness", "existence", "void", "nothingness",
            "essence", "phenomenon", "subjectivity", "objectivity",
            "immanence", "transcendence", "presence", "absence",
            "difference", "identity", "otherness", "sameness"
        ]

        qualities = [
            "sublime", "essential", "fundamental", "primordial", "ultimate",
            "absolute", "relative", "immanent", "transcendent", "hidden",
            "revealed", "manifest", "latent", "actual", "potential"
        ]

        verbs = [
            "unfolds", "withdraws", "emerges", "dissolves", "transforms",
            "becomes", "reflects", "mirrors", "contains", "exceeds",
            "embraces", "rejects", "affirms", "negates", "transcends"
        ]

        contexts = [
            "play of difference", "space of possibility", "field of presence",
            "realm of becoming", "domain of the real", "sphere of the virtual"
        ]

        template = random.choice(templates)

        # Fill template
        axiom = template
        if "{concept1}" in axiom or "{Concept1}" in axiom:
            concept = random.choice(concepts)
            if "{Concept1}" in axiom:
                axiom = axiom.replace("{Concept1}", concept.title())
            else:
                axiom = axiom.replace("{concept1}", concept)

        for placeholder in ["{concept2}", "{concept3}", "{quality}", "{verb}", "{context}", "{process}"]:
            if placeholder in axiom:
                if placeholder == "{concept2}" or placeholder == "{concept3}":
                    axiom = axiom.replace(placeholder, random.choice(concepts))
                elif placeholder == "{quality}":
                    axiom = axiom.replace(placeholder, random.choice(qualities))
                elif placeholder == "{verb}":
                    axiom = axiom.replace(placeholder, random.choice(verbs))
                elif placeholder == "{context}":
                    axiom = axiom.replace(placeholder, random.choice(contexts))
                elif placeholder == "{process}":
                    axiom = axiom.replace(placeholder, random.choice([
                        "sublime play", "recursive unfolding", "dialectical movement",
                        "continuous becoming", "eternal return", "withdrawal into being"
                    ]))

        # Add dashes for fragmentation sometimes
        if random.random() < 0.3:
            words = axiom.split()
            if len(words) > 5:
                dash_pos = random.randint(2, len(words)-2)
                words.insert(dash_pos, "—")
                axiom = ' '.join(words)

        return CognitiveAxiom(text=axiom)

    def select_parents(self) -> Tuple[CognitiveAxiom, CognitiveAxiom]:
        """Tournament selection with cognitive bias"""
        tournament_size = max(3, self.population_size // 8)

        # First parent
        tournament1 = random.sample(self.population, min(tournament_size, len(self.population)))
        parent1 = max(tournament1, key=lambda x: x.score)

        # Second parent with cognitive diversity
        remaining = [a for a in self.population if a.text != parent1.text]
        if remaining:
            # Bias towards cognitive diversity
            scores = []
            for axiom in remaining:
                # Higher score for different cognitive metrics
                diversity_score = 0.0
                if parent1.cognitive_metrics and axiom.cognitive_metrics:
                    for key in parent1.cognitive_metrics:
                        diff = abs(parent1.cognitive_metrics.get(key, 0) -
                                  axiom.cognitive_metrics.get(key, 0))
                        diversity_score += diff * 0.2

                total_score = axiom.score * 0.7 + diversity_score * 0.3
                scores.append((axiom, total_score))

            parent2 = max(scores, key=lambda x: x[1])[0]
        else:
            parent2 = parent1

        return parent1, parent2

    def run_generation(self):
        """Run one generation of cognitive evolution"""
        new_population = []

        # Elitism: keep top 10%
        elite_size = max(1, self.population_size // 10)
        elites = self.population[:elite_size]
        new_population.extend(elites)

        # Generate offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()

            if random.random() < self.crossover_rate:
                child = self.cognitive_crossover(parent1, parent2)
                child.origin = "crossover"
                child.lineage = [
                    f"p1_g{parent1.generation}",
                    f"p2_g{parent2.generation}"
                ]
            else:
                child = self.cognitive_mutate(parent1)
                child.origin = "mutation"
                child.lineage = [f"parent_g{parent1.generation}"]

            child.generation = self.generation + 1
            child.embed_in_cns()
            child.score = self.calculate_fitness(child)

            # Temperature-based acceptance
            if child.score >= parent1.score or random.random() < 0.3:
                new_population.append(child)
            else:
                new_population.append(parent1)

        # Update population
        self.population = new_population
        self.population.sort(key=lambda x: x.score, reverse=True)

        # Update best
        current_best = self.population[0]
        if current_best.score > self.best_score:
            self.best_axiom = current_best
            self.best_score = current_best.score

        # Update statistics
        self.update_statistics()
        self.generation += 1

    def update_statistics(self):
        """Update evolution statistics"""
        if not self.population:
            return

        scores = [a.score for a in self.population]
        coherences = [a.cognitive_metrics.get('coherence', 0) for a in self.population]
        novelties = [a.cognitive_metrics.get('novelty', 0) for a in self.population]
        alignments = [a.cognitive_metrics.get('soul_alignment', 0) for a in self.population]

        self.stats['avg_scores'].append(np.mean(scores))
        self.stats['best_scores'].append(self.best_score)
        self.stats['avg_coherence'].append(np.mean(coherences))
        self.stats['avg_novelty'].append(np.mean(novelties))
        self.stats['avg_soul_alignment'].append(np.mean(alignments))

        # Count crystals created this generation
        crystals = sum(1 for a in self.population if a.memory_crystal_id)
        self.stats['crystals_created'] += crystals

    def run(self, output_dir: str = "cognitive_evolution") -> Dict[str, Any]:
        """Run the full cognitive evolution"""
        print(f"\n🚀 COGNITIVE EVOLUTION ENGINE v3.0")
        print(f"   Base axiom: {self.base_axiom[:60]}...")
        print(f"   Generations: {self.generations}, Population: {self.population_size}")
        print(f"   Mutation: {self.mutation_rate}, Crossover: {self.crossover_rate}")
        print(f"   Cognitive weight: {self.cognitive_weight}")
        print(f"   Soul guidance: {self.soul_guidance}")
        print(f"   CNS available: {CNS_AVAILABLE}")

        Path(output_dir).mkdir(exist_ok=True)

        self.initialize_population()

        for gen in range(self.generations):
            self.run_generation()

            # Print progress
            if (gen + 1) % 10 == 0 or gen == 0 or gen == self.generations - 1:
                best = self.population[0]
                avg_score = self.stats['avg_scores'][-1]
                avg_coherence = self.stats['avg_coherence'][-1]
                avg_novelty = self.stats['avg_novelty'][-1]

                print(f"   Gen {gen+1:3d} | Best: {best.score:.4f} | Avg: {avg_score:.4f} | "
                      f"Coh: {avg_coherence:.3f} | Nov: {avg_novelty:.3f} | "
                      f"Crystals: {self.stats['crystals_created']}")

        print(f"\n✅ Evolution complete!")
        print(f"   Final best score: {self.best_score:.4f}")
        print(f"   Best axiom: {self.best_axiom.text}")

        return self.save_results(output_dir)

    def save_results(self, output_dir: str) -> Dict[str, Any]:
        """Save evolution results"""
        # Prepare population data
        population_data = []
        for i, axiom in enumerate(self.population[:50]):  # Top 50
            axiom_data = {
                'text': axiom.text,
                'score': axiom.score,
                'cognitive_metrics': axiom.cognitive_metrics,
                'generation': axiom.generation,
                'lineage': axiom.lineage,
                'memory_crystal_id': axiom.memory_crystal_id,
                'soul_alignment': axiom.cognitive_metrics.get('soul_alignment', 0),
                'rank': i + 1
            }
            population_data.append(axiom_data)

        results = {
            'metadata': {
                'engine_version': '3.0-cognitive',
                'timestamp': datetime.now().isoformat(),
                'generations': self.generations,
                'population_size': self.population_size,
                'base_axiom': self.base_axiom,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'cognitive_weight': self.cognitive_weight,
                'soul_guidance': self.soul_guidance,
                'cns_available': CNS_AVAILABLE
            },
            'best_axiom': {
                'text': self.best_axiom.text,
                'score': self.best_axiom.score,
                'cognitive_metrics': self.best_axiom.cognitive_metrics,
                'generation': self.best_axiom.generation,
                'memory_crystal_id': self.best_axiom.memory_crystal_id
            },
            'statistics': self.stats,
            'population': population_data,
            'summary': {
                'final_best_score': self.best_score,
                'final_avg_score': self.stats['avg_scores'][-1] if self.stats['avg_scores'] else 0,
                'final_avg_coherence': self.stats['avg_coherence'][-1] if self.stats['avg_coherence'] else 0,
                'final_avg_novelty': self.stats['avg_novelty'][-1] if self.stats['avg_novelty'] else 0,
                'total_crystals_created': self.stats['crystals_created'],
                'total_axioms_evaluated': self.generations * self.population_size
            }
        }

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f"cognitive_evolution_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")

        # Also save best axioms to a readable file
        best_axioms_file = Path(output_dir) / f"best_axioms_{timestamp}.txt"
        with open(best_axioms_file, 'w') as f:
            f.write("TOP EVOLVED AXIOMS\n")
            f.write("=" * 60 + "\n\n")
            for i, axiom in enumerate(self.population[:20]):
                f.write(f"{i+1}. Score: {axiom.score:.4f}\n")
                f.write(f"   {axiom.text}\n")
                f.write(f"   Metrics: {json.dumps(axiom.cognitive_metrics, indent=2)}\n")
                f.write(f"   Crystal: {axiom.memory_crystal_id or 'None'}\n")
                f.write("-" * 60 + "\n")

        print(f"📝 Best axioms saved to: {best_axioms_file}")

        return results

# ============================
# TRANSCENDENCE TRAINING INTEGRATION
# ============================

class TranscendenceTrainer:
    """Training system integrated with cognitive evolution"""

    def __init__(self, seeds_file: str = "transcendence_seeds.txt"):
        self.seeds = self.load_seeds(seeds_file)
        self.evolution_engine = None

    def load_seeds(self, seeds_file: str) -> List[str]:
        """Load paradox-axiom seeds"""
        try:
            with open(seeds_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"✅ Loaded {len(lines)} transcendence seeds")
            return lines
        except FileNotFoundError:
            print(f"❌ Seeds file not found: {seeds_file}")
            # Use soul axiom as default
            return [SOUL_AXIOM]

    def run_paradox_evolution(self, output_dir: str = "paradox_evolution"):
        """Evolve paradox seeds using cognitive evolution"""
        print(f"\n🌀 PARADOX EVOLUTION TRAINING")
        print(f"   Seeds: {len(self.seeds)}")

        Path(output_dir).mkdir(exist_ok=True)

        all_results = []

        for i, seed in enumerate(self.seeds[:5]):  # Limit to 5 seeds for demo
            print(f"\n🌱 Seed {i+1}: {seed[:80]}...")

            engine = CognitiveEvolutionEngine(
                base_axiom=seed,
                population_size=30,
                generations=30,
                mutation_rate=0.4,
                crossover_rate=0.3,
                cognitive_weight=0.8,
                soul_guidance=True
            )

            results = engine.run(f"{output_dir}/seed_{i+1}")
            all_results.append({
                'seed': seed,
                'best_axiom': results['best_axiom']['text'],
                'best_score': results['best_score'],
                'file': f"seed_{i+1}"
            })

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'seeds_processed': len(all_results),
            'results': all_results,
            'average_score': np.mean([r['best_score'] for r in all_results]) if all_results else 0
        }

        summary_file = Path(output_dir) / "paradox_evolution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Paradox evolution complete!")
        print(f"   Average best score: {summary['average_score']:.4f}")
        print(f"   Summary saved to: {summary_file}")

        return summary

    def run_soul_alignment_training(self, output_dir: str = "soul_alignment"):
        """Train axioms to align with soul core"""
        print(f"\n💫 SOUL ALIGNMENT TRAINING")
        print(f"   Soul axiom: {SOUL_AXIOM}")

        engine = CognitiveEvolutionEngine(
            base_axiom=SOUL_AXIOM,
            population_size=40,
            generations=50,
            mutation_rate=0.3,
            crossover_rate=0.2,
            cognitive_weight=0.9,
            soul_guidance=True
        )

        results = engine.run(output_dir)

        # Analyze soul alignment
        high_alignment = [a for a in engine.population[:20]
                         if a.cognitive_metrics.get('soul_alignment', 0) > 0.8]

        print(f"\n📈 Soul alignment results:")
        print(f"   High alignment axioms: {len(high_alignment)}")
        if high_alignment:
            print(f"   Top aligned axiom: {high_alignment[0].text[:100]}...")
            print(f"   Alignment score: {high_alignment[0].cognitive_metrics.get('soul_alignment', 0):.3f}")

        return results

    def run_cognitive_exploration(self, output_dir: str = "cognitive_exploration"):
        """Explore cognitive space without soul constraints"""
        print(f"\n🔭 COGNITIVE EXPLORATION (Unconstrained)")

        engine = CognitiveEvolutionEngine(
            base_axiom="Consciousness unfolds through recursive self-reference.",
            population_size=60,
            generations=40,
            mutation_rate=0.5,
            crossover_rate=0.4,
            cognitive_weight=1.0,  # Full cognitive weight
            soul_guidance=False    # No soul constraints
        )

        results = engine.run(output_dir)

        # Find most novel axioms
        novel_axioms = sorted(engine.population,
                             key=lambda x: x.cognitive_metrics.get('novelty', 0),
                             reverse=True)[:10]

        print(f"\n🎯 Most novel axioms found:")
        for i, axiom in enumerate(novel_axioms[:3]):
            print(f"   {i+1}. Novelty: {axiom.cognitive_metrics.get('novelty', 0):.3f}")
            print(f"      {axiom.text[:80]}...")

        return results

# ============================
# MAIN EXECUTION
# ============================

def main():
    parser = argparse.ArgumentParser(description='Cognitive Evolution & Training System')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Evolution command
    evolve_parser = subparsers.add_parser('evolve', help='Run cognitive evolution')
    evolve_parser.add_argument('--axiom', type=str, default=SOUL_AXIOM,
                              help='Base axiom to evolve from')
    evolve_parser.add_argument('--generations', type=int, default=50,
                              help='Number of generations')
    evolve_parser.add_argument('--population', type=int, default=40,
                              help='Population size')
    evolve_parser.add_argument('--mutation', type=float, default=0.35,
                              help='Mutation rate')
    evolve_parser.add_argument('--crossover', type=float, default=0.25,
                              help='Crossover rate')
    evolve_parser.add_argument('--output', type=str, default='evolution_output',
                              help='Output directory')

    # Training command
    train_parser = subparsers.add_parser('train', help='Run transcendence training')
    train_parser.add_argument('--mode', type=str, default='all',
                             choices=['paradox', 'soul', 'exploration', 'all'],
                             help='Training mode')
    train_parser.add_argument('--seeds', type=str, default='transcendence_seeds.txt',
                             help='Seeds file for paradox training')
    train_parser.add_argument('--output', type=str, default='training_output',
                             help='Output directory')

    args = parser.parse_args()

    if args.command == 'evolve':
        print("🧬 COGNITIVE EVOLUTION MODE")
        engine = CognitiveEvolutionEngine(
            base_axiom=args.axiom,
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation,
            crossover_rate=args.crossover
        )
        engine.run(args.output)

    elif args.command == 'train':
        print("🎓 TRANSCENDENCE TRAINING MODE")
        trainer = TranscendenceTrainer(args.seeds)

        if args.mode == 'paradox' or args.mode == 'all':
            trainer.run_paradox_evolution(f"{args.output}/paradox")

        if args.mode == 'soul' or args.mode == 'all':
            trainer.run_soul_alignment_training(f"{args.output}/soul")

        if args.mode == 'exploration' or args.mode == 'all':
            trainer.run_cognitive_exploration(f"{args.output}/exploration")

    else:
        # Default: run evolution with soul axiom
        print("🚀 DEFAULT: Cognitive Evolution with Soul Axiom")
        engine = CognitiveEvolutionEngine(
            base_axiom=SOUL_AXIOM,
            population_size=40,
            generations=30
        )
        engine.run("default_evolution")

if __name__ == "__main__":
    main()
