#!/usr/bin/env python3
"""
Modular Spectral Transport Framework with Data-Driven Parameter Justification for SU(4)
--------------------------------------------------------------------------------------------

This version configures representation input parameters for SU(4) and calculates:
  1. The spectral (mass) gap.
  2. The mass-energy equivalence of the gap using E = m c².

Canonical SU(4) representations considered:
    - "1": Trivial representation (degree = 1)
    - "F": Fundamental representation (degree = 4)      [assigned prime = 5]
    - "6": Two-index antisymmetric representation (degree = 6) [assigned prime = 7]
    - "A": Adjoint representation (degree = 15)           [assigned prime = 11]

The prime adjacency is computed as:
    adjacency(p) = 1 + 1/√p
which encodes spectral tension based on a prime number.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable
from abc import ABC, abstractmethod


# ---------- Representation Module ----------

class Representation(ABC):
    """Abstract base class for a group representation."""

    @abstractmethod
    def get_identifier(self) -> str:
        """Return a unique identifier for the representation."""
        pass


class FiniteRepresentation(Representation):
    """
    Concrete implementation for a finite group representation.

    Attributes:
        identifier (str): Unique label.
        degree (int): Dimension of the irreducible representation.
        prime_adjacent (float): Weight computed from a prime index, reflecting spectral tension.
    """

    def __init__(self, identifier: str, degree: int, prime_adjacent: float):
        self.identifier = identifier
        self.degree = degree
        self.prime_adjacent = prime_adjacent

    def get_identifier(self) -> str:
        return self.identifier


# ---------- Data-Driven Parameter Justification ----------

# Lookup table for representation degrees and associated prime values for SU(4).
REPRESENTATION_LOOKUP = {
    "SU(4)": {
        "representations": {
            "1": {"degree": 1, "prime": None},  # Trivial representation
            "F": {"degree": 4, "prime": 5},  # Fundamental representation
            "6": {"degree": 6, "prime": 7},  # Two-index antisymmetric representation
            "A": {"degree": 15, "prime": 11}  # Adjoint representation
        }
    }
}


def compute_prime_adjacency(prime: int) -> float:
    """
    Compute prime adjacency using the function:
      adjacency(p) = 1 + 1/√p
    """
    return 1 + 1 / math.sqrt(prime)


def configure_representation(lie_group: str, identifier: str) -> FiniteRepresentation:
    """
    Configure a FiniteRepresentation instance for a given Lie group and identifier.
    Uses a lookup table to get the canonical degree and associated prime,
    then computes the prime adjacency weight.
    """
    group_data = REPRESENTATION_LOOKUP.get(lie_group)
    if not group_data or identifier not in group_data["representations"]:
        raise ValueError(f"Representation '{identifier}' not defined for group '{lie_group}'.")
    rep_data = group_data["representations"][identifier]
    degree = rep_data["degree"]
    prime = rep_data["prime"]
    prime_adjacent = compute_prime_adjacency(prime) if prime else 1.0
    return FiniteRepresentation(identifier, degree, prime_adjacent)


# ---------- Skyrmion Charge Lattice Module ----------

class SkyrmionChargeLattice:
    """
    Maps a representation to its quantized wrapping energy (topological energy cost).

    The wrapping energy is computed as:
        minimal_gap * (degree * prime_adjacent)
    """

    def __init__(self, minimal_gap: float = 0.001):
        self.minimal_gap = minimal_gap

    def wrapping_energy(self, rep: Representation) -> float:
        if rep.get_identifier() == "1":
            return 0.0  # trivial representation
        else:
            return self.minimal_gap * rep.degree * rep.prime_adjacent


# ---------- Modular Transport Operator Module ----------

class ModularTransportOperator:
    """
    Represents the McKay-type modular transport operator.
    Computes eigenvalues as the sum of a base eigenvalue with a wrapping-energy weight.
    """

    def __init__(self, representations: List[Representation],
                 weight_func: Callable[[Representation], float]):
        self.representations = representations
        self.weight_func = weight_func

    def compute_operator(self) -> Dict[str, float]:
        base_value = 3  # Simulated base eigenvalue (e.g., from a Laplacian operator)
        eigenvalues = {}
        for rep in self.representations:
            weight = self.weight_func(rep)
            eigenvalues[rep.get_identifier()] = base_value + weight
        return eigenvalues


# ---------- Spectral Pinning Engine Module ----------

class SpectralPinningEngine:
    """
    Computes the spectral (mass) gap, defined as the difference between
    the vacuum eigenvalue and the smallest non-vacuum eigenvalue.
    """

    def __init__(self, eigenvalues: Dict[str, float]):
        self.eigenvalues = eigenvalues

    def compute_gap(self) -> float:
        if "1" not in self.eigenvalues:
            raise ValueError("Trivial representation '1' must be included in the eigenvalue set.")
        vacuum_value = self.eigenvalues["1"]
        nontrivial_values = [val for key, val in self.eigenvalues.items() if key != "1"]
        gap = min(nontrivial_values) - vacuum_value
        return gap


# ---------- Mass-Energy Equivalence Utility ----------

def mass_energy_equivalence(mass: float, speed_of_light: float = 299792458) -> float:
    """
    Convert a mass (or mass gap) to its energy equivalent using E = m c².

    Args:
        mass (float): The mass (or mass gap).
        speed_of_light (float): The speed of light in m/s. Default is ~299,792,458 m/s.

    Returns:
        float: The energy equivalent in Joules.
    """
    return (mass * speed_of_light ** 2)


# ---------- Visualization Utility ----------

def plot_eigenvalues(eigenvalues: dict, group_name: str):
    identifiers = list(eigenvalues.keys())
    values = list(eigenvalues.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(identifiers, values, color='skyblue')
    plt.xlabel('Representation Identifier')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalue Spectrum for {group_name}')

    # Highlight the vacuum (trivial representation)
    for bar, rep_id in zip(bars, identifiers):
        if rep_id == "1":
            bar.set_color('salmon')
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     'Vacuum', ha='center', va='bottom')
    plt.show()


# ---------- Main Simulation ----------

def main():
    lie_group = "SU(4)"  # Setting the gauge group to SU(4)

    # Configure representations for SU(4) via our lookup approach.
    rep_trivial = configure_representation(lie_group, "1")
    rep_F = configure_representation(lie_group, "F")
    rep_6 = configure_representation(lie_group, "6")
    rep_A = configure_representation(lie_group, "A")
    representations = [rep_trivial, rep_F, rep_6, rep_A]

    # Initialize the Skyrmion Charge Lattice with a chosen minimal gap.
    skyrmion_lattice = SkyrmionChargeLattice(minimal_gap=1.602e-16) #1MeV =1.602e-13, e-17 is Python FPL & breaks calc
    #need to import mpmath or decimal handle the ultrafine resolution or sub-spectral analysis one would imagine
    def weight_function(rep: Representation) -> float:
        return skyrmion_lattice.wrapping_energy(rep)

    # Create the modular transport operator.
    transport_operator = ModularTransportOperator(representations, weight_function)
    eigenvalues = transport_operator.compute_operator()

    print("Computed Eigenvalues for Each Representation:")
    for rep_id, val in eigenvalues.items():
        print(f"  Representation {rep_id}: Eigenvalue = {val:.40f}")

    # Compute the spectral (mass) gap.
    pinning_engine = SpectralPinningEngine(eigenvalues)
    gap = pinning_engine.compute_gap()
    print(f"\nComputed Spectral (Mass) Gap: Δ = {gap:.40f}")

    # Express the gap in terms of mass-energy equivalence.
    energy_gap = mass_energy_equivalence(gap)
    print(f"Mass-Energy Equivalence: E = {energy_gap:.20e} Joules")

    # Plot eigenvalues for visual feedback.
    plot_eigenvalues(eigenvalues, lie_group)


if __name__ == "__main__":
    main()
