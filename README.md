# Cognitive Multi-Domain EW–Guidance Co-Designer (COG-EW)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B%20(CPU%2FGPU)-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![RadioML 2018.01A](https://img.shields.io/badge/Dataset-RadioML%202018.01A%20(24--Class)-yellow.svg)](https://www.deepsig.ai/datasets)
[![Deployment](https://img.shields.io/badge/Render-Cloud%20Ready-46E3B7.svg)](https://render.com/)
[![Defense AI](https://img.shields.io/badge/Domain-Aerospace%20%26%20Defense-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Autonomous Cognitive Electronic Warfare, Deep Residual RF Perception, and Closed-Loop Hypersonic Guidance System**

| Attribute | Details |
|---|---|
| **Operational Domain** | Aerospace & Defense – Cognitive AI / Electromagnetic Spectrum Operations (EMSO) |
| **System Level** | Production-Grade AI/ML System Architecture & Mission Control Platform |
| **Target Platform** | Hypersonic Strike Vehicles (Mach 5.2) / FastAPI Mission Engine / Antigravity IDE |
| **Core Methodologies** | 24-Class Deep ResNet AMC, True Proportional Navigation, Cognitive RGPO / Noise Jamming |
| **Benchmark Standard** | RadioML 2018.01A (60/10/30 Locked Evaluation across 766,772 Test Samples) |

---

## 1. Executive Summary & Core Philosophy

This repository presents a unified, production-ready implementation of the **Cognitive Multi-Domain EW–Guidance Co-Designer (COG-EW)**. Engineered for hypersonic vehicles penetrating heavily defended, highly contested anti-access/area-denial (A2/AD) airspace, the system autonomously coordinates two tightly coupled operational domains in real time:

*   **Electronic Warfare (EW):** Continuously monitors enemy RF radar and communication emissions, classifies modulation signatures using a 24-class Deep Residual Network in under 3 milliseconds, and autonomously commands adaptive countermeasures (passive monitoring, barrage noise saturation, or Range Gate Pull-Off deceptive spoofing).
*   **Guidance & Trajectory Planning:** Controls vehicle flight dynamics (angle of attack, bank angle, Mach 5.2 cruise, and terminal dive) to strike strategic targets on schedule while actively evading Surface-to-Air Missile (SAM) intercepts.

### Core Paradigm: Multi-Domain Co-Design
Traditional military platforms treat flight guidance and electronic warfare as disconnected sub-systems. **COG-EW jointly optimizes both domains**: flight maneuvers modify radar cross-section (RCS) illumination and Doppler geometry, while electronic deception degrades missile tracking solutions, enabling coordinated, high-survivability strike missions.

---

## 2. Problem Statement & Mission Context

Modern Integrated Air Defense Systems (IADS) deploy multi-static, frequency-agile radar networks paired with supersonic SAM interceptors. Traditional platforms suffer from:
1. **Reaction Latency:** Human operators and legacy EW pods cannot analyze complex digital modulations and execute evasive maneuvers within hypersonic millisecond engagement windows.
2. **Decoupled Control Penalties:** High-G physical evasion without electronic jamming bleeds excessive kinetic energy, while jamming without kinetic maneuver allows track-via-missile (TVM) home-on-jam weapons to score direct hits.
3. **Severe Operational Constraints:** Hypersonic flight imposes extreme thermal heating bounds, aerodynamic structural limits ($\le 12G$), and rapid Mach 5+ descent dynamics.

---

## 3. System Architecture & Workflow Diagram

The COG-EW architecture links real-time radio frequency signal intelligence directly to aerodynamic flight guidance and autonomous tactical countermeasures:

```mermaid
flowchart TB
    subgraph RF_ENV["1. Electromagnetic Spectrum & RF Threat Network"]
        IADS["Enemy Multi-Static IADS Radars<br/>(VHF Search / S-Band Acquisition / X-Band Fire Control)"]
        PULSE["Intercepted Raw I/Q Pulses<br/>[Shape: 2 x 1024 Float32]"]
        IADS -->|"Electromagnetic Emission"| PULSE
    end

    subgraph PERCEPTION["2. Perception & Signal Intelligence (perception/)"]
        direction TB
        RESNET["24-Class Deep ResNet AMC<br/>(resnet_best.pt | Sub-3ms Latency)"]
        DSP["Signal Quality & Spectral DSP Engine<br/>(FFT PSD, I/Q Imbalance, DC Offset)"]
        FUSION["Threat Assessment & Situation Synthesis<br/>(Search vs Tracking vs Missile Lock)"]

        PULSE --> RESNET
        PULSE --> DSP
        RESNET -->|"Modulation Probabilities"| FUSION
        DSP -->|"Estimated SNR & Signal Integrity"| FUSION
    end

    subgraph COGNITIVE["3. Autonomous Decision & Control AI (rl/ & guidance/)"]
        direction TB
        DECISION["Hierarchical Cognitive Commander<br/>(Multi-Objective Optimization)"]
        SAFETY["Kinematic & Thermal Safety Filter<br/>(12G Structural & Heating Bounds)"]
        FUSION --> DECISION
        DECISION --> SAFETY
    end

    subgraph DUAL_ACTION["4. Coordinated Multi-Domain Execution"]
        direction LR
        subgraph EW_DOMAIN["Electronic Warfare Domain"]
            PASSIVE["ESM Radio Silence"]
            NOISE["Barrage Noise Jamming<br/>(45 kW Saturation)"]
            RGPO["Range Gate Pull-Off (RGPO)<br/>(False Phantom Echo Deception)"]
        end

        subgraph FLIGHT_DOMAIN["Guidance & Flight Domain"]
            CRUISE["Hypersonic Glide<br/>(Mach 5.2 | Alt 29,500m)"]
            EVADE["High-G Bank-to-Turn Evasion<br/>(6.8G Pull | Lock Break)"]
            DIVE["Terminal Supersonic Dive<br/>(Kinetic Bunker Strike)"]
        end
    end

    SAFETY -->|"EW Directives"| EW_DOMAIN
    SAFETY -->|"Flight Controls"| FLIGHT_DOMAIN

    subgraph SIM_BATTLE["5. Tactical Combat Engine (simulation/ & sim/)"]
        direction TB
        SAM["Enemy SAM Interceptor<br/>(Mach 4.0 Proportional Nav)"]
        BUNKER["Strategic Target Command Bunker<br/>(Sector 4 Objective)"]

        RGPO -->|"Doppler False Target"| SAM
        EVADE -->|"Kinematic Separation"| SAM
        SAM -->|"Missile Lock Broken"| BURNOUT["SAM Motor Exhaustion / Defeat"]
        DIVE -->|"Direct Kinetic Hit"| BUNKER
    end

    subgraph TELEMETRY["6. Mission Control & Presentation Layer (dashboard/ & scripts/)"]
        DASH_API["FastAPI Unified API (scripts/dashboard_api.py)"]
        UI1["Tactical Battlefield Radar Map (60 FPS Canvas)"]
        UI2["RF Waveform & AI Softmax Oscilloscope"]
        UI3["Spectral FFT & Signal Quality Gauges"]
        UI4["Locked 60/10/30 Benchmark & SNR Stratification"]

        SIM_BATTLE -.->|"Real-Time State Telemetry"| DASH_API
        DASH_API --> UI1
        DASH_API --> UI2
        DASH_API --> UI3
        DASH_API --> UI4
    end

    classDef env fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef ai fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef decision fill:#311042,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef action fill:#022c22,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef sim fill:#3f1d0b,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef dash fill:#172554,stroke:#60a5fa,stroke-width:2px,color:#f8fafc;

    class RF_ENV env;
    class PERCEPTION ai;
    class COGNITIVE decision;
    class DUAL_ACTION action;
    class SIM_BATTLE sim;
    class TELEMETRY dash;
```

### 3.1 Closed-Loop Mission Execution Sequence

The operational lifecycle of an autonomous engagement follows a verified closed-loop sequence:

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Mission Commander / Pilot
    participant Veh as Hypersonic Vehicle (Mach 5.2)
    participant RF as RF Perception (Deep ResNet)
    participant Decision as Cognitive Decision Engine
    participant Jammer as Cognitive Jammer (RGPO)
    participant EnemyRadar as Enemy IADS Radar Network
    participant SAM as Surface-to-Air Missile (Mach 4)
    participant Bunker as Strategic Bunker (Sector 4)

    Operator->>Veh: Launch Mission (Mach 5.2, 29,500m Altitude)
    EnemyRadar->>Veh: Sector 1 Search Radar Sweep (8ASK/FM)
    Veh->>RF: Intercept RF Waveform (2x1024 I/Q)
    RF->>Decision: Emitter ID: Early Warning Search (Low Threat)
    Decision->>Veh: Maintain Stratospheric Cruise (Radio Silence)

    EnemyRadar->>Veh: Sector 3 X-Band Fire-Control Lock (16QAM)
    Veh->>RF: Intercept Fire-Control Radar Pulses
    RF->>Decision: Threat Level: CRITICAL (Missile Guidance Active)
    EnemyRadar->>SAM: Launch Interceptor Missile at Vehicle

    Decision->>Jammer: Engage Range Gate Pull-Off (RGPO) Spoofing
    Jammer-->>SAM: Transmit Amplified Doppler False Echo
    Decision->>Veh: Execute 6.8G Evasive Turn Maneuver
    SAM->>Jammer: Locks onto False Phantom Target
    Note over SAM: SAM Diverts, Over-shoots & Suffers Motor Burnout

    Decision->>Veh: Initiate Terminal Supersonic Dive Profile
    Veh->>Bunker: High-Speed Kinetic Impact at Coordinates
    Note over Bunker: Target Eliminated (100% Mission Accomplished)
```

### 3.2 Simulation Environment Module (`simulation/` & `sim/`)
*   **Hypersonic Vehicle Dynamics:** High-fidelity 60 FPS flight kinematics, aerodynamic lift/drag, gravity, atmospheric density, and strict constraints (max 12G structural load, stagnation thermal heating in kJ, altitude profile from 29,500m to 250m).
*   **RF Threat Environment:** Multi-static IADS radar network featuring Sector 1 Early Warning Search (VHF/UHF), Sector 2 Target Acquisition (S-Band), and Sector 3 Fire-Control (X-Band pencil beam).
*   **EW Effects Model:** Simulates jamming noise-floor elevation (reducing radar detection range) and RGPO spoofing track-degradation (inducing Doppler phantom targets and SAM missile break-locks).

### 3.3 Perception & Signal Intelligence Module (`perception/`)
*   **RF Signal Preprocessing:** High-speed I/Q ingestion from real datasets (`D:/Datasets/RadioML 2018.01A`) or high-fidelity mathematical pulse synthesis.
*   **Emitter Classification Network:** 24-class Deep Residual Network (`perception/models/radioml_resnet.py`) with residual skip connections delivering sub-3ms inference latency and >94% operational accuracy ($\ge +6\text{ dB}$).
*   **Signal Quality & Spectral DSP:** Real-time computation of FFT Power Spectral Density, estimated SNR, I/Q imbalance, DC offset, and crest factor.

### 3.4 Decision & Control AI Module (`rl/` & `guidance/`)
*   **POMDP Formulation:** 
    *   State ($s$): Vehicle kinematic state + threat graph + intercepted RF modulation probabilities + SAM proximity.
    *   Action ($a$): EW directives (Passive / Noise Jamming / RGPO Spoofing) + flight heading and bank angle.
    *   Reward ($r$): Multi-objective Pareto score balancing survivability, target destruction, thermal limits, and energy.
*   **Guidance Laws:** True Proportional Navigation Guidance (PNG) combined with autonomous high-G evasive maneuvers and terminal supersonic strike profiling.
*   **Safety & Explainability:** Action projection safety filters enforce aerodynamic limits, while explanation generators provide human-readable tactical rationales.

---

## 4. Advanced Research Upgrade Roadmap (17 Frontier Features)

To address real limitations in autonomous electronic warfare and guidance systems, 17 high-value research upgrades have been integrated into the co-designer framework:

1. **Hierarchical Cognitive RL**
   * *Problem:* One RL policy controls all sub-domains simultaneously.
   * *Solution:* Strategic Commander AI coordinating Guidance, EW, and Resource specialists.
   * *Novelty:* Modular expert collaboration and conflict resolution.
2. **Adaptive Enemy Intelligence**
   * *Problem:* Static or scripted simulator behavior leads to brittle AI.
   * *Solution:* Enemy radars learn and change frequency, scan patterns, and cooperation across missions.
   * *Novelty:* AI vs AI adversary training instead of AI vs scripted simulator.
3. **Cognitive Battlefield Understanding**
   * *Problem:* Simple reactive decision-making based on immediate feature vectors.
   * *Solution:* Build a Threat Knowledge Graph and infer enemy intent before acting.
   * *Novelty:* Deeper situational understanding instead of simple reaction.
4. **Memory-Augmented Tactical AI**
   * *Problem:* Inability to retain operational experience across multiple mission phases.
   * *Solution:* Store previous encounters, successful jamming strategies, and radar behavior in long-term memory.
   * *Novelty:* Long-term tactical memory and retrieval-augmented decision-making.
5. **Dynamic Mission Intelligence**
   * *Problem:* Rigid reward functions fail when mission parameters shift mid-flight.
   * *Solution:* Mission objectives can change during execution (attack, return, recon, divert).
   * *Novelty:* Dynamic reward adaptation and objective switching.
6. **Multi-Objective Decision Engine**
   * *Problem:* Single-scalar rewards blur trade-offs between competing physical goals.
   * *Solution:* Optimize mission success, survivability, fuel, heat, jammer power, and time simultaneously.
   * *Novelty:* Realistic multi-objective Pareto trade-off optimization.
7. **Explainable Tactical Intelligence**
   * *Problem:* 'Black-box' neural policies create distrust in defense applications.
   * *Solution:* Every action includes a human-readable explanation and confidence score.
   * *Novelty:* Trustworthy AI with self-generating decision rationale.
8. **Predictive Battlefield Intelligence**
   * *Problem:* Reactive control causes lag in taking defensive countermeasures.
   * *Solution:* Predict future radar states and risk trajectories before taking action.
   * *Novelty:* Proactive look-ahead planning instead of reactive response.
9. **Risk Assessment Engine**
   * *Problem:* High-risk maneuvers executed without evaluating failure probability.
   * *Solution:* Estimate detection, missile lock, thermal failure, and mission failure risks in real time.
   * *Novelty:* Quantified decision support and probabilistic risk modeling.
10. **Self-Evolving Reward System**
    * *Problem:* Hardcoded rewards cause policy stagnation during complex training.
    * *Solution:* Reward priorities evolve dynamically through curriculum training stages.
    * *Novelty:* Adaptive curriculum-driven reward shaping.
11. **Physics-Aware AI**
    * *Problem:* Post-processing safety filters often discard optimal learned policies.
    * *Solution:* Physics constraints directly influence learning representation and loss functions.
    * *Novelty:* Co-design of AI learning objectives and vehicle flight dynamics.
12. **Confidence-Aware AI**
    * *Problem:* Overconfident AI actions under high sensor noise or signal spoofing.
    * *Solution:* Estimate uncertainty (epistemic and aleatoric) before executing aggressive actions.
    * *Novelty:* Risk-sensitive, uncertainty-guided decision-making under ambiguity.
13. **Decentralized Collaborative Intelligence**
    * *Problem:* Swarm failure when communications are severed in contested environments.
    * *Solution:* Autonomous agents cooperate effectively despite communication loss.
    * *Novelty:* Robust decentralized autonomous teamwork.
14. **Battlefield Digital Twin**
    * *Problem:* AI overfitting to fixed simulation environment layouts.
    * *Solution:* Generate diverse, realistic synthetic battlefields for robust training.
    * *Novelty:* High-fidelity generalization across unseen geographical and RF topographies.
15. **Human-AI Collaboration**
    * *Problem:* Full autonomy lacks operational human oversight.
    * *Solution:* AI recommends multiple tactical strategies with clear explanations to the human operator.
    * *Novelty:* Flexible human-in-the-loop and human-on-the-loop autonomy.
16. **Intent Prediction Engine**
    * *Problem:* Inability to discern whether an enemy radar is passively scanning or targeting.
    * *Solution:* Predict whether enemies are searching, tracking, preparing missiles, or deceiving.
    * *Novelty:* Real-time enemy intent inference using temporal neural models.
17. **Cognitive Knowledge Graph**
    * *Problem:* Standard vector representations lose relational context between emitters and terrain.
    * *Solution:* Represent entities and spatial/RF relationships in a graph neural framework.
    * *Novelty:* Relational reasoning for electromagnetic tactical environments.

---

## 5. Recommended Publication Architecture & Top Contributions

### Integrated Cognitive Pipeline Architecture

```mermaid
flowchart TD
    OBJ["Strategic Mission Objective & Target Tasking"] --> CMD["Strategic Cognitive Commander"]

    subgraph SPECIALISTS["Domain Expert Specialists"]
        direction LR
        G_SPEC["Guidance Specialist<br/>(Trajectory & Aerodynamics)"]
        EW_SPEC["EW Specialist<br/>(Jamming & Deception)"]
        RES_SPEC["Resource Specialist<br/>(Thermal & Power Management)"]
    end

    CMD --> G_SPEC
    CMD --> EW_SPEC
    CMD --> RES_SPEC

    G_SPEC --> MO_OPT["Multi-Objective Decision Engine<br/>(Pareto Trade-off: Survivability vs Target Arrival)"]
    EW_SPEC --> MO_OPT
    RES_SPEC --> MO_OPT

    MO_OPT --> SAFETY["Physics & Safety Projection Layer<br/>(Aerodynamic G-limits, Heat Dissipation, Stall Margins)"]
    SAFETY --> ACTION["Executed Multi-Domain Tactical Action"]

    ACTION --> TWIN["Battlefield Digital Twin Simulation<br/>(Kinematics, Emitters, Missiles)"]

    TWIN --> ADAPT["Adaptive Enemy AI + Threat Knowledge Graph"]
    ADAPT --> MEMORY["Long-Term Tactical Experience Memory"]
    MEMORY --> INTENT["Enemy Intent Prediction & Risk Engine"]
    INTENT --> XAI["Explainable AI (XAI) Tactical Rationale Generator"]
    XAI -.->|"Feedback & Lifelong Learning"| CMD

    classDef cmdNode fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#ffffff;
    classDef specNode fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef optNode fill:#311042,stroke:#c084fc,stroke-width:2px,color:#ffffff;
    classDef actNode fill:#022c22,stroke:#34d399,stroke-width:2px,color:#ffffff;

    class CMD,OBJ cmdNode;
    class G_SPEC,EW_SPEC,RES_SPEC specNode;
    class MO_OPT,SAFETY optNode;
    class ACTION,TWIN,ADAPT,MEMORY,INTENT,XAI actNode;
```

### Top Recommended Research Contributions
| Rank | Contribution | Publication Value | Difficulty |
|:---:|---|:---:|:---:|
| 1 | **Hierarchical Cognitive Commander** | ★★★★★ | High |
| 2 | **Adaptive Enemy Intelligence** | ★★★★★ | Medium |
| 3 | **Multi-Objective Decision Engine** | ★★★★★ | Medium |
| 4 | **Intent Prediction Engine** | ★★★★★ | High |
| 5 | **Long-Term Tactical Memory** | ★★★★☆ | Medium |
| 6 | **Explainable Tactical Intelligence** | ★★★★☆ | Medium |

---

## 6. Detailed Codebase Directory & File Structure

```
COG-EW/
├── README.md                          # Comprehensive technical documentation & workflow diagrams
├── requirements.txt                   # Production & Render cloud dependencies (Lightweight CPU PyTorch)
├── requirements-dev.txt               # Full local development & training dependencies (Gym, SB3, MLflow)
├── render.yaml                        # Automated cloud deployment blueprint for Render Web Services
├── train.py                           # Baseline training entry point
├── evaluate.py                        # Benchmark evaluation script
├── configs/                           # System Configuration Files
│   ├── env_config.yaml                # Gym & hypersonic environment parameters
│   └── train_config.yaml              # RL agent & PPO hyperparameters
├── dashboard/                         # Unified Mission Control & Presentation Layer
│   ├── index.html                     # Unified Mission Control Cockpit (Overview redirect)
│   ├── pages/                         # Core Mission Intelligence Interfaces
│   │   ├── overview.html              # Tactical overview, 60 FPS battlefield canvas & controls
│   │   ├── rf_classification.html     # Live 24-class RF modulation oscilloscope & threat meter
│   │   ├── signal_analysis.html       # Spectral FFT PSD, constellation & signal quality gauges
│   │   └── model_performance.html     # Locked test benchmarks & interactive SNR threshold filter
│   └── assets/                        # High-performance styling (CSS) & procedural audio assets
├── scripts/                           # Core Operational & Service Scripts
│   ├── dashboard_api.py               # Unified FastAPI backend & tactical mission server
│   ├── train_radioml_finalists.py     # 60/10/30 RadioML multi-seed training pipeline
│   └── run_radioml_cv_tuning.py       # Cross-validation & hyperparameter optimization engine
├── simulation/                        # High-Fidelity Tactical Combat Engine
│   ├── tactical_engine.py             # 60 FPS hypersonic aerodynamics (Mach 5.2), IADS radars & SAMs
│   └── demo_engine.py                 # Interactive demonstration & scenario coordinator
├── perception/                        # RF Perception & Signal Intelligence
│   ├── rf_inference_service.py        # 24-class AMC inference service & DSP signal quality analyzer
│   └── models/                        # Deep Neural Network Architectures
│       ├── radioml_resnet.py          # 24-Class Deep ResNet with residual skip connections
│       └── radioml_cnn.py             # Baseline 1D/2D CNN architectures
├── models/                            # Trained Checkpoints
│   └── experiments/radioml_60_10_30_v1/
│       └── resnet_lr2.5e-4_wd5e-4_best.pt # Winning trained 24-class ResNet (6.42 MB)
├── reports/                           # Locked Benchmark Reports & Confusion Matrices
│   └── experiments/radioml_60_10_30_v1/
│       └── finalist_comparison_report.json # Comprehensive 766,772 test sample evaluation
├── sim/                               # POMDP Gym Simulation Environment
│   ├── vehicle_dynamics.py            # 2D/3D flight kinematics, aero & thermal limits
│   ├── emitter_models.py              # Multi-static IADS radar layout & signal emission
│   └── ew_effects.py                  # Jamming noise-floor & spoofing error models
├── guidance/                          # Guidance & Dynamic Utilities
│   ├── baseline_guidance.py           # True Proportional Navigation Law (PNG) baseline
│   └── dynamics_utils.py              # Coordinate frame transformations (NED, ECEF)
├── rl/                                # Decision & Control AI Module
│   ├── policy_network.py              # Multi-head PPO/MAPPO actor-critic network
│   ├── reward.py                      # Multi-objective Pareto reward engine
│   └── hierarchical_commander.py      # Strategic Commander coordinating domain experts
├── safety/                            # Safety Projection Filter
│   └── constraint_checker.py          # Kinematic/thermal/g-load action projection filter
├── explain/                           # Explainability & Interpretability
│   └── explanation_generator.py       # Human-readable tactical rationale generator
└── tests/                             # Comprehensive Automated Test Suite
    ├── test_dashboard_api.py          # FastAPI endpoints & static file serving tests
    ├── test_tactical_and_rf_system.py # End-to-end tactical simulation & RF perception tests
    └── test_decision_engine.py        # Autonomous decision & EW jamming logic tests
```

---

## 7. Training Strategy & Comprehensive Evaluation Metrics

### 7.1 Data & Training Strategy
*   **Synthetic Data Generation:** RF pulse patterns and IADS network configurations are synthetically generated within the simulator across diverse geographic layouts.
*   **Curriculum Learning:** Agent training begins in sparse environments (1–3 static emitters) and progressively scales to dense, networked, adaptive radar topologies.
*   **Meta-Learning & Lifelong Adaptation:** Employs meta-RL techniques to enable rapid policy adaptation when encountering novel, unseen enemy radar configurations, backed by continuous post-mission policy updates.

### 7.2 Evaluation Metrics & Performance Benchmarks
*   **Mission Success Rate:** Percentage of trajectories reaching designated target within specified time-on-target and positioning accuracy thresholds.
*   **Survivability Index:** Integrated cumulative detection probability over time and total probability of kill ($P_k$) reduction.
*   **Trajectory Quality:** Smoothness, physical feasibility, and strict adherence to structural/thermal limits (zero constraint violations).
*   **EW Effectiveness:** Quantified reduction in enemy tracking lock-on duration and total radar track breaks achieved.
*   **Computational Efficiency:** Inference latency per decision cycle maintained strictly under **50 ms** to ensure real-time execution.

---

## 8. Implementation Roadmap (Phases)

*   **Phase 0: Setup & Orientation:** Initialize structure in Antigravity IDE, configure Python environment (Python 3.10+, PyTorch, NumPy, SciPy), setup Git repo, and define team roles.
*   **Phase 1: Dynamics & Baseline Guidance:** Build 2D point-mass vehicle model, aerodynamics, thermal constraints, and baseline Proportional Navigation guidance tracker.
*   **Phase 2: RF Threat & EW Environment:** Implement radar emitter pulse generation, multi-static coverage geometry, jamming noise elevation, and spoofing models.
*   **Phase 3: Perception & Classification:** Train 1D CNN / Transformer classifier on synthetic pulse data; build real-time situation summary generator and Threat Knowledge Graph.
*   **Phase 4: RL Formulation & Baseline Agent:** Implement Gym environment, multi-head PPO agent, and train baseline joint EW-guidance policy.
*   **Phase 5: Advanced RL & Meta-Learning:** Incorporate curriculum learning, multi-agent PPO (MAPPO), Hierarchical Cognitive Commander, and scenario adaptation experiments.
*   **Phase 6: Safety & Explainability:** Integrate constraint safety filter, attention visualizers, and explainable decision generator for interpretable tactical decision-making.
*   **Phase 7: Evaluation & Final Report:** Run comprehensive benchmarks against baselines, generate analytical charts, publish final report, and validate 50ms inference constraints.

---

## 9. Risk Analysis & Mitigation Strategy

*   **Risk 1: High Computational Complexity**
    * *Mitigation:* First build a Minimum Viable Product (MVP) with simplified 2D dynamics and static emitters before introducing complex, multi-agent networked IADS.
*   **Risk 2: RL Non-Convergence in Multi-Domain Space**
    * *Mitigation:* Carefully design reward components, implement curriculum learning, use self-evolving reward shaping, and perform systematic hyperparameter tuning.
*   **Risk 3: Time Overrun on Advanced Modules**
    * *Mitigation:* Enforce rigid phase milestones, conduct weekly check-ins, and focus strictly on core deliverables before enabling optional research extensions.

---

## 10. Getting Started & Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/pravatsahu05/COG-EW.git
cd COG-EW

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate

# Install production dependencies (lightweight CPU PyTorch stack)
pip install -r requirements.txt

# (Optional) For deep RL training, MLflow, and full research tools:
pip install -r requirements-dev.txt
```

### Usage

```bash
# Launch the Unified Tactical Mission Control & RF Intelligence API
.\.venv\Scripts\python.exe -m uvicorn scripts.dashboard_api:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
- **Tactical Mission Control (Home):** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Tactical Combat Simulation Console:** [http://127.0.0.1:8000/pages/simulation.html](http://127.0.0.1:8000/pages/simulation.html)
- **RF 24-Class Modulation Intelligence Console:** [http://127.0.0.1:8000/pages/rf_classification.html](http://127.0.0.1:8000/pages/rf_classification.html)
- **Signal Quality & Spectral FFT Analysis:** [http://127.0.0.1:8000/pages/signal_analysis.html](http://127.0.0.1:8000/pages/signal_analysis.html)
- **Model Performance & Operational SNR Stratification:** [http://127.0.0.1:8000/pages/model_performance.html](http://127.0.0.1:8000/pages/model_performance.html)
- **Interactive Swagger REST API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 11. Core Objectives: Implementation & Verification Summary

COG-EW delivers a complete, cohesive end-to-end framework addressing all four core requirements:

```mermaid
flowchart LR
    subgraph OBJ1["1. RF Signal Perception & Jamming"]
        direction TB
        O1_A["24-Class RadioML ResNet AMC"] --> O1_B["Threat Level Assessment<br/>(Search / Track / SAM Lock)"]
        O1_B --> O1_C["Adaptive Jamming Trigger<br/>(Noise Barrage vs RGPO Deception)"]
    end

    subgraph OBJ2["2. Auto-Pilot & SAM Evasion"]
        direction TB
        O2_A["SAM Launch Alert Detection"] --> O2_B["RGPO Phantom Target Echo"]
        O2_B --> O2_C["6.8G Evasive Split-S Maneuver"]
        O2_C --> O2_D["Missile Break-Lock & Burnout"]
    end

    subgraph OBJ3["3. Tactical Gaming Console"]
        direction TB
        O3_A["60 FPS Canvas Radar Display"] --> O3_B["Procedural Web Audio Engine"]
        O3_B --> O3_C["Multi-Scenario Mission Presets"]
        O3_C --> O3_D["Automated Victory Debrief Modal"]
    end

    subgraph OBJ4["4. Strategic Bunker Strike"]
        direction TB
        O4_A["Terminal Supersonic Dive Profile"] --> O4_B["Sector 4 Command Bunker Strike"]
        O4_B --> O4_C["Direct Kinetic Impact Confirmed"]
        O4_C --> O4_D["Exportable JSON Telemetry Log"]
    end

    OBJ1 ==> OBJ2 ==> OBJ3 ==> OBJ4

    classDef obj1 fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#ffffff;
    classDef obj2 fill:#022c22,stroke:#34d399,stroke-width:2px,color:#ffffff;
    classDef obj3 fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#ffffff;
    classDef obj4 fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#ffffff;

    class OBJ1 obj1;
    class OBJ2 obj2;
    class OBJ3 obj3;
    class OBJ4 obj4;
```

### Dual-Layer Architecture: Judge Presentation vs Real Production Execution
COG-EW is engineered with a strict two-layer architecture:
1. **Judge Presentation & Demonstration Layer (`dashboard/pages/simulation.html`):**
   - High-fidelity visual cockpit and radar sweep battlefield designed specifically for non-technical evaluation, visual inspection, and interactive demonstrations.
2. **Real Production Execution Layer (`perception/`, `sim/`, `guidance/`, `safety/`, `scripts/`):**
   - Autonomous, mathematically rigorous Python execution engine performing real aerothermodynamic heating integration, radar Swerling detection probability, True Proportional Navigation, and Deep Learning inference on **actual radar datasets stored in `D:/Datasets`**.

### Objective 1: Reading Radar Signals and Jamming Them
- **Perception Engine (`perception/rf_inference_service.py`):**
  - Connects directly to the real dataset directory `D:/Datasets` (specifically `D:\Datasets\RadioML 2018.01A ⭐ Main RF benchmark\GOLD_XYZ_OSC.0001_1024.hdf5` and local locked test splits).
  - Evaluates actual $(2, 1024)$ I/Q radar pulses through the trained 24-class PyTorch ResNet model (`models/experiments/radioml_60_10_30_v1/resnet_lr2.5e-4_wd5e-4_best.pt`).
  - Computes exact signal quality metrics (estimated SNR, I/Q imbalance, DC offset, crest factor) and fast Fourier transform (FFT) Power Spectral Density.
  - Maps modulation predictions to tactical operational threat levels:
    - **Search / Early Warning:** `OOK`, `4ASK`, `8ASK`, `FM` &rarr; Low threat &rarr; Passive monitoring / low-power spot noise.
    - **Tracking Radars (T-1):** `BPSK`, `QPSK`, `8PSK` &rarr; High threat &rarr; Active Barrage Noise Jamming (72% radar suppression).
    - **Fire-Control SAM Radars (M-1):** `16QAM`, `32QAM`, `64QAM` &rarr; Critical threat &rarr; Range Gate Pull-Off (RGPO) Deceptive Spoofing.


### Objective 2: Auto-Pilot System for Missile Defense & Evasion
- **Autonomous Guidance & Evasion Engine (`simulation/tactical_engine.py`):**
  - **Proportional Navigation Guidance:** Maintains optimal hypersonic trajectory toward strategic coordinates while monitoring radar threat envelopes.
  - **SAM Threat Intercept Trigger:** Detects missile launches from enemy fire-control batteries (`dist < 220 km` warning perimeter).
  - **Cognitive Coordinated Countermeasure:**
    1. **Electromagnetic:** Automatically activates high-power RGPO deceptive spoofing (42.5 kW), synthesizing a Doppler-shifted false phantom target echo 160 units away.
    2. **Kinematic:** Autonomous high-G evasive maneuver ($6.8g$, $45^\circ$ bank angle) perpendicular to missile vector.
  - **Result:** Enemy SAM breaks lock, tracks the false phantom echo into empty space, and suffers motor burnout or premature detonation harmlessly away from the vehicle.

### Objective 3: Gaming Simulation Console to Present to Judges
- **Interactive Mission Console (`dashboard/pages/simulation.html`):**
  - **Visual Presentation:** 60 FPS HTML5 Canvas tactical radar screen featuring rotating radar scan beams, dynamic range rings, vehicle shockwave cones, missile smoke trails, and cyan/purple jamming emission waves.
  - **Zero-Dependency Web Audio Synthesis:** Procedural audio generating radar blips, missile launch alert sirens, electronic jamming hums, and cinematic target detonation explosions.
  - **Scenario Presets:**
    1. *Coastal Penetration:* Early warning search radar crossing.
    2. *Contested Airspace:* Search radar + Fire-control SAM battery.
    3. *Deep Multi-Layered Strike:* Full IADS battle with Search (S-1), Tracking (T-1), and SAM Battery (M-1).
  - **Operator Interactivity:** Start/Pause, scenario selection, manual flight override, auto-pilot toggle, and manual emergency evasion trigger.
  - **Judges Debrief Modal:** Automatic pop-up upon mission completion displaying survivability rating, SAMs defeated, radar sites suppressed, and an instant JSON flight telemetry download button.

### Objective 4: Hitting and Destroying the Assigned Target
- **Strategic Target Strike Logic:**
  - Designated target: **Enemy Strategic Radar Command Bunker** located at Sector 4 `(x: 1080.0, y: 325.0, radius: 28.0)`.
  - Upon clearing the radar defense perimeter, the auto-pilot transitions from ingress glide to a terminal supersonic dive trajectory directly into the bunker coordinates.
  - Direct kinetic impact is verified geometrically; triggers the impact explosion particle system, logs the event `TARGET_DESTROYED`, updates the mission status to `100% ACCOMPLISHED`, and generates the comprehensive mission debrief report.

---

## 12. Judge Demonstration Guide & Viva Talking Points

When presenting COG-EW to academic evaluators or project judges, follow this sequence:

1. **Start the FastAPI Backend:**
   ```powershell
   cd D:\COG-EW
   .\.venv\Scripts\python.exe -m uvicorn scripts.dashboard_api:app --host 127.0.0.1 --port 8000
   ```
2. **Open the Simulation Console:**
   Navigate to [http://127.0.0.1:8000/pages/simulation.html](http://127.0.0.1:8000/pages/simulation.html) and unmute the sound using the speaker button.
3. **Select Scenario 3 ("Deep Multi-Layered Strike"):**
   Click **"Engage Mission / Start"**.
4. **Point Out During Flight:**
   - **RF Emitter Detection:** Show the right-hand HUD as Search Radar S-1 illuminates the craft; note the modulation class (`8ASK`) and passive monitoring response.
   - **SAM Launch & Missile Defense:** As the vehicle enters SAM Battery M-1's range, note the warning siren, the missile launch (`SAM-1`), the automatic switch to **RGPO Spoofing**, and the vehicle's $6.8g$ evasive turn breaking the missile lock.
   - **Target Destruction:** Watch the vehicle execute its terminal dive, directly striking and eliminating the Command Bunker with particle explosions.
5. **Review the Debrief Report:**
   Examine the automatically opened Victory Modal showing 100% mission accomplishment, 0% vehicle damage, and click **"Export Flight Telemetry"** to inspect the structured mission log.

