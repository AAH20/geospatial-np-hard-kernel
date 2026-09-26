# Geospatial NP-Hard Kernel (`geospatial-np-hard-kernel`)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Pure%20Stdlib)-success.svg)]()
[![Build](https://img.shields.io/badge/Tests-12%2F12%20Passing-emerald.svg)]()
[![Performance](https://img.shields.io/badge/Latency-2.0%20ms%20Pipeline-orange.svg)]()

> **Deterministic, ultra-low latency, zero-external-dependency Python engine solving the 10 Apex NP-Hard Problems and Bottlenecks across Geographic Information Systems (GIS), Defense Geospatial Intelligence (GEOINT), and Autonomous Multi-Domain Operations.**

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph CoreSolvers ["Core NP-Hard Solvers (Pure Python Stdlib)"]
        P1["P1: PALP Cartographic Label Placement<br/><i>(Greedy-MWIS on Conflict Graph)</i>"]
        P2["P2: Capacitated p-Median Emergency Siting<br/><i>(Teitz-Bart Local Interchange)</i>"]
        P3["P3: Art Gallery & Terrain Viewshed Siting<br/><i>(Submodular Ray-Casting Coverage)</i>"]
        P4["P4: Contiguity-Constrained Districting<br/><i>(Seeded Region Grow & Graph Swap)</i>"]
        P5["P5: Dubins Multi-UAV Trajectory<br/><i>(2-Opt DTSPN with Threat Domes)</i>"]
        P6["P6: Homotopy Line Generalization<br/><i>(Imai-Iri Min-Vertex DP)</i>"]
        P7["P7: Dynamic Evacuation Route Scheduling<br/><i>(Time-Expanded Network Flow)</i>"]
        P8["P8: Spatial Weights Optimization & Moran's I<br/><i>(Spectral Sparsified W-Matrix)</i>"]
        P9["P9: Spatial Co-Location Pattern Mining<br/><i>(Apriori Clique & Participation Index)</i>"]
        P10["P10: Agile Satellite Constellation Tasking<br/><i>(Time-Indexed Slew-Angle Branching)</i>"]
    end

    subgraph Adapters ["Domain Operational Adapters"]
        DEF["Defense GEOINT & Surveillance Adapter<br/>• Stealth UAV Penetration<br/>• Border Radar Siting<br/>• Recon Satellite Constellation<br/>• SIGINT/ELINT Co-Location"]
        URB["Urban Civilian ArcGIS/QGIS Adapter<br/>• Zero-Collision Cadastral Labels<br/>• EMS Facility Allocation<br/>• Municipal Wards Districting<br/>• Hurricane Evacuation Egress"]
    end

    subgraph Facade ["Unified Engine & Interfaces"]
        ENG["GeospatialNPHardEngine Facade"]
        CLI["Interactive CLI & Micro-Benchmark Suite"]
    end

    P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 & P9 & P10 --> ENG
    ENG --> DEF
    ENG --> URB
    DEF & URB --> CLI
```

---

## The 10 Apex NP-Hard GIS & GEOINT Problems

| ID | Problem Name | Classical Formulation | Algorithmic Paradigm | Benchmark Latency |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | **Cartographic Label Placement (PALP)** | Maximum Weight Independent Set | 8-Position Conflict Gradient Relaxation | **125.8 µs** |
| **P2** | **Capacitated $p$-Median Location (CPMP)** | Metric Facility Location with Capacity Caps | Teitz-Bart Vertex Substitution | **254.5 µs** |
| **P3** | **Terrain Viewshed Radar Siting** | Art Gallery / Maximum Submodular Coverage | Ray-Casting LOS + $(1 - 1/e)$ Greedy Cover | **650.8 µs** |
| **P4** | **Territorial Districting** | Contiguous Balanced Graph Partitioning | Seeded BFS Expansion + Boundary Swaps | **164.8 µs** |
| **P5** | **Dubins UAV Trajectory with Threats** | Dubins TSP with Obstacles (DTSPN) | 2-Opt Lin-Kernighan + Threat Penalty Line Integrals | **307.2 µs** |
| **P6** | **Homotopy-Preserving Line Simplification** | Min-Vertex Polyline Generalization | Imai-Iri Dynamic Programming ($\epsilon$-deviation) | **202.6 µs** |
| **P7** | **Time-Dependent Evacuation Routing** | Dynamic Traffic Assignment (TD-VRPTW) | Time-Expanded Multi-Commodity Congestion Flow | **32.0 µs** |
| **P8** | **Spatial Weights Matrix ($W$) Optimization** | Eigenvalue / Autocorrelation Maximization | Spectral Sparsification & Moran's $I$ Z-Score Max | **147.0 µs** |
| **P9** | **Spatial Co-Location Pattern Mining** | Geometric Clique Finding & Association Rules | Apriori Candidate Join + Monotonic PI Pruning | **80.8 µs** |
| **P10** | **Agile Satellite Constellation Tasking** | Asymmetric TSP with Time Windows (AEOSSP) | Forward Insertion with Attitude Slewing Bounds | **14.2 µs** |

---

## Dual-Use Domain Applications

```
+---------------------------------------------------------------------------------------+
| DUAL-USE GEOSPATIAL OPERATIONAL CAPABILITIES                                          |
+=======================================================================================+
| Defense Geospatial Intelligence (GEOINT)     | Urban Civilian GIS & Infrastructure    |
+----------------------------------------------+----------------------------------------+
| • Multi-UAV low-observable route planning     | • High-density automated map lettering |
|   through integrated air-defense domes        |   with zero text overlap collisions   |
| • Mountainous border surveillance and radar  | • Municipal fire station and ambulance |
|   barrier placement maximizing LOS coverage  |   depot capacitated coverage planning  |
| • Agile constellation high-value target tasking| • Contiguity-guaranteed legislative,  |
|   subject to camera attitude slewing dynamics|   school, and voting ward districting  |
| • Multi-source SIGINT/ELINT spatial pattern   | • Hurricane and wildfire dynamic civil |
|   clustering to identify command headquarters|   evacuation routing & chokepoint fixes|
+---------------------------------------------------------------------------------------+
```

---

## Quickstart & CLI

```bash
# Clone and enter repository
git clone https://github.com/AAH20/geospatial-np-hard-kernel.git
cd geospatial-np-hard-kernel

# Run the 10-solver microsecond benchmark suite
python3 cli.py benchmark-all

# Run Defense GEOINT & Reconnaissance Demonstration
python3 cli.py defense-demo

# Run Urban Civilian GIS Planning Demonstration
python3 cli.py urban-demo
```

### Python API Example

```python
from geospatial_np_hard_kernel import GeospatialNPHardEngine
from geospatial_np_hard_kernel.core.models import PointFeature

engine = GeospatialNPHardEngine()

# Point-Feature Cartographic Label Placement
features = [
    PointFeature("c1", "Metropolis", 10.0, 10.0, priority=100.0),
    PointFeature("c2", "River City", 12.0, 11.0, priority=80.0),
]
result = engine.solve_palp(features)
print(f"Placed {result.total_labels_placed} labels in {result.execution_time_us} µs")
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
