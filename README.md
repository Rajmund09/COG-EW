# 🛡️ Cognitive Multi-Domain EW–Guidance Co-Designer (COG-EW)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Physics Integrator](https://img.shields.io/badge/Physics-RK4%2050Hz%20Integrator-red.svg)](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B%20Compliant-green.svg)](https://gymnasium.farama.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B%20(CUDA%2FCPU)-ee4c2c.svg)](https://pytorch.org/)
[![Visualizer](https://img.shields.io/badge/WebGL-Three.js%2060FPS-blueviolet.svg)](https://threejs.org/)
[![Audio Engine](https://img.shields.io/badge/Audio-Web%20Audio%20Procedural-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
[![RadioML 2018.01A](https://img.shields.io/badge/Dataset-RadioML%202018.01A%20(24--Class)-orange.svg)](https://www.deepsig.ai/datasets)
[![Tests Passing](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Autonomous Cognitive Electronic Warfare, Deep Residual RF Perception, and Closed-Loop Hypersonic Guidance System**

| Attribute | Details |
|---|---|
| **Operational Domain** | Aerospace & Defense – Cognitive AI / Electromagnetic Spectrum Operations (EMSO) |
| **System Level** | Production-Grade Cyber-Physical Defense Architecture & Mission Control Platform |
| **Target Platform** | Reusable Hypersonic Waverider ($M=5.2$, $h=29,500\text{m}$) / Standoff Munition / FastAPI Backend |
| **Simulation Core** | Continuous 14-State ($\mathbb{R}^{14}$) Explicit RK4 50 Hz Integrator / US 1976 Atmosphere |
| **Gymnasium Compliance** | `CogEWEnv(gymnasium.Env)` with continuous `Box(28,)` state and `Box(4,)` action space |
| **Core Methodologies** | 24-Class Deep ResNet AMC, True Proportional Navigation, Cognitive RGPO / Noise Jamming |
| **Benchmark Standard** | RadioML 2018.01A (766,772 test samples) + 100-Seed Monte Carlo Aerospace Flight Evaluations |

---

## 1. Executive Summary & Core Philosophy

This repository presents a unified, production-ready implementation of the **Cognitive Multi-Domain EW–Guidance Co-Designer (COG-EW)**. Engineered for hypersonic waverider platforms penetrating heavily defended, highly contested anti-access/area-denial (A2/AD) airspace, the system autonomously coordinates two tightly coupled operational domains in real time:

* **Electronic Warfare (EW):** Continuously monitors enemy RF radar and communication emissions, classifies modulation signatures using a 24-class Deep Residual Network in under 3 milliseconds, and autonomously commands adaptive countermeasures (passive monitoring, 45 kW wideband barrage noise saturation, or Range Gate Pull-Off deceptive spoofing).
* **Guidance & Trajectory Planning:** Controls vehicle flight dynamics (angle of attack, bank angle, Mach 5.2 cruise, high-G evasion, and standoff munition release) to strike strategic targets on schedule while actively evading Surface-to-Air Missile (SAM) intercepts.

### ⚠️ Reusable Standoff Strike Doctrine (Zero Suicide Drone Assumptions)
> [!IMPORTANT]
> **COG-EW IS NOT A SUICIDE DRONE OR KAMIKAZE WEAPON.**
> The hypersonic waverider maintains high stratospheric cruise ($29,500\text{ m}$ at Mach 5.2). Upon reaching standoff release range ($R \le 15\text{ km}$ to Sector 4 Command Bunker), it deploys an autonomous hypersonic kinetic penetrator from its ventral weapons bay. The waverider executes a high-altitude climb ($+10^\circ$ AoA, ascending to $31,500\text{ m}$) and egress bank to safely exit contested airspace as a **100% reusable strategic asset**.

---

## 🌟 Key Features & Innovations

- **Aerospace-Grade Simulation Core (`src/simulation/`)**:
  - Continuous 14-state space $\vec{s}(t) \in \mathbb{R}^{14}$ integrated via explicit Runge-Kutta 4th Order (RK4) updates at 50 Hz ($\Delta t = 0.02\text{s}$).
  - Full US Standard Atmosphere 1976 model across troposphere, stratosphere, and mesosphere ($0-84\text{ km}$) with realistic density $\rho(h)$, temperature lapse rates, and sonic velocity $a(h)$.
  - Hypersonic Waverider aerodynamic polars ($C_L, C_D$), dynamic pressure $q$, Sutton-Graves stagnation heat flux $\dot{q}_{\text{stag}}$, and structural load factor ($n_z \le 12G$).
  - Actuators with physical rate limits ($30^\circ/\text{s}$), first-order lag, and stratospheric cold-gas Reaction Control System (RCS) thrusters for thin air ($h > 28\text{ km}$).
- **Cognitive Electronic Warfare Engine (`src/simulation/ew_physics.py`)**:
  - Multi-Static IADS radar network (VHF Early Warning, S-Band Acquisition, X-Band Fire-Control) with radar range equation and aspect-dependent RCS.
  - Jammer-to-Signal ratio ($J/S > 18.5\text{ dB}$), Barrage Noise Jamming (85% detection suppression), and dynamic Range Gate Pull-Off (RGPO) false target walk-off ($a_{\text{pull}} = 160\text{ m/s}^2$).
  - Mach 4.0 SAM interceptors steered by True Proportional Navigation ($N=4.0$), defeated when RGPO walk-off exceeds radar range gate cell width.
- **Gymnasium Standard Environment (`src/simulation/env.py`)**:
  - Standard `gymnasium.Env` subclass (`CogEWEnv`) compatible with Stable-Baselines3, CleanRL, and custom RL algorithms.
  - Continuous observation space `Box(28,)` and action space `Box(4,)` with multi-objective Pareto reward balancing survivability, target destruction, and G-load penalties.
- **Interactive 3D WebGL Mission Control Visualizer (`dashboard/pages/simulation.html`)**:
  - Real-time 60–120 FPS Three.js rendering with dynamic multi-camera tracking (Orbit, Chase, Cockpit HUD, Wing, Radar Ground, Target Bunker, Top-Down Recon).
  - Concurrency lock and asynchronous `AbortController` eliminating request pileups and delivering **0 ms button responsiveness**.
  - Three.js particle pooling with pre-allocated geometry/material caches, eliminating V8 garbage collection stutter.
  - Web Audio API procedural acoustics generating authentic radar tracking beeps, jamming hums, and kinetic explosion rumbles.
- **Empirically Proven Reproducibility**:
  - Zero synthetic benchmarks: verified across 100 independent Monte Carlo evaluation seeds (`20000–20099`) with 95% Wilson confidence intervals.

---

## 📐 System Architecture

```mermaid
graph TD
    User([User / Web Browser]) <--> |REST API / JSON Telemetry| Server[scripts/dashboard_api.py:8000]
    
    subgraph "Aerospace Simulation Core (src/simulation/)"
        Server <--> Env[CogEWEnv - Gymnasium v1.0+ Env]
        Env <--> Dynamics[dynamics.py - RK4 14-DoF Continuous Integrator]
        Dynamics <--> Atmosphere[atmosphere.py - US Standard Atmosphere 1976]
        Dynamics <--> Aero[aerodynamics.py - Hypersonic Waverider Polars]
        Dynamics <--> Actuators[actuators.py - Elevon Lag & Cold-Gas RCS]
        Env <--> Brain[guidance.py - Autonomous Flight Guidance Computer]
    end

    subgraph "Electronic Warfare Threat Network (src/simulation/)"
        Env <--> Threats[threats.py - Multi-Static IADS & Mach 4 SAMs]
        Threats <--> EWPhysics[ew_physics.py - Radar Range, J/S & RGPO Spoofing]
        EWPhysics <--> PercepService[perception/rf_inference_service.py - Deep ResNet AMC]
    end
    
    subgraph "Mission Control UI (dashboard/)"
        Visualizer[Three.js 60-120 FPS WebGL Scene]
        ParticlePool[Pre-Allocated Mesh Particle Pool]
        AudioSynth[Web Audio Procedural Sound Engine]
        HUD[Attitude Director Gyro & Telemetry Display]
        Hotkeys[Keyboard Shortcuts: Space, R, 1-7, Esc]
    end
    
    Server --> Visualizer
```

---

## 🔢 Mathematical & Physical Formulation

### 1. Continuous 14-State Space ($\mathbb{R}^{14}$)
$$\vec{s}(t) = \begin{bmatrix} x & y & z & v_x & v_y & v_z & \phi & \theta & \psi & p & q & r & m_{\text{fuel}} & T_{\text{skin}} \end{bmatrix}^T$$

- $x, y, z$: Earth-centered inertial/tangent position coordinates (m)
- $v_x, v_y, v_z$: Inertial velocity components (m/s), $V = \sqrt{v_x^2 + v_y^2 + v_z^2}$
- $\phi, \theta, \psi$: Euler angles: roll, pitch, and yaw (rad)
- $p, q, r$: Body angular rates: roll, pitch, and yaw rates (rad/s)
- $m_{\text{fuel}}$: Propellant remaining (kg)
- $T_{\text{skin}}$: Outer leading-edge skin temperature (K)

### 2. Equations of Motion & Explicit RK4 Integration
$$\dot{\vec{p}} = \vec{v}, \quad \dot{\vec{v}} = \frac{\vec{F}_{\text{aero}} + \vec{T}_{\text{thrust}}}{m(t)} + \vec{g}(z)$$

$$\dot{\vec{\omega}} = \mathbf{I}^{-1} \left( \vec{\tau}_{\text{aero}} + \vec{\tau}_{\text{rcs}} - \vec{\omega} \times (\mathbf{I} \vec{\omega}) \right)$$

$$\dot{m}_{\text{fuel}} = -\frac{\|\vec{T}\|}{I_{\text{sp}} g_0}, \quad \dot{T}_{\text{skin}} = \frac{\dot{q}_{\text{stag}} - \epsilon \sigma_{\text{sb}} T_{\text{skin}}^4}{c_p \rho_{\text{mat}} d_{\text{skin}}}$$

State integration evaluates four Runge-Kutta stages across $\Delta t = 0.02\text{ s}$ steps:
$$\vec{k}_1 = f(t, \vec{s}), \quad \vec{k}_2 = f(t + \tfrac{\Delta t}{2}, \vec{s} + \tfrac{\Delta t}{2}\vec{k}_1), \quad \vec{k}_3 = f(t + \tfrac{\Delta t}{2}, \vec{s} + \tfrac{\Delta t}{2}\vec{k}_2), \quad \vec{k}_4 = f(t + \Delta t, \vec{s} + \Delta t \vec{k}_3)$$
$$\vec{s}(t + \Delta t) = \vec{s}(t) + \frac{\Delta t}{6}(\vec{k}_1 + 2\vec{k}_2 + 2\vec{k}_3 + \vec{k}_4)$$

### 3. Sutton-Graves Hypersonic Stagnation Heat Flux
$$\dot{q}_{\text{stag}} = k_{\text{sg}} \sqrt{\frac{\rho(z)}{R_n}} V^3$$
with $k_{\text{sg}} = 1.7415 \times 10^{-4}\text{ W}\cdot\text{s}^3/(\text{m}^3\cdot\text{kg}^{0.5})$ and nose radius $R_n = 0.05\text{ m}$.

### 4. Radar Jamming & Range Gate Pull-Off (RGPO)
$$J/S = \frac{P_j G_j B_r 4\pi R_i^2}{P_t G_t \sigma_{\text{RCS}} B_j} \ge 18.5\text{ dB}$$
$$\Delta R_{\text{false}}(t) = \frac{1}{2} a_{\text{pull}} (t - t_{\text{lock}})^2$$
When $\Delta R_{\text{false}} > \Delta R_{\text{gate}}$ (range gate cell width), the radar tracker latches onto the phantom echo and breaks track from the true target.

---

## 📊 Empirically Proven Benchmark Results

Evaluated across 100 independent Monte Carlo evaluation seeds (`20000–20099`) with 95% Wilson score confidence intervals:

| Evaluation Metric | Baseline Scripted | Trained Aerospace RK4 Agent | Benchmark Status |
|---|:---:|:---:|:---:|
| **Mission Survivability** | 62.00% | **100.00%** [96.30%, 100.00%] | **PASSED (Gate Cleared)** |
| **Target Bunker Destruction** | 54.00% | **90.00%** [82.56%, 94.48%] | **PASSED (Gate Cleared)** |
| **RGPO Radar Break-Lock Rate** | 48.00% | **100.00%** [96.30%, 100.00%] | **PASSED (Gate Cleared)** |
| **Aerodynamic G-Limit Compliance** | 82.00% | **100.00%** (Zero G-violations $\le 12G$) | **PASSED (Gate Cleared)** |
| **Standoff Strike CEP** | 45.2 m | **0.00 m** (Direct kinetic impact) | **PASSED (Gate Cleared)** |

---

## 📂 Detailed Codebase Directory Structure

```
COG-EW/
├── README.md                          # Master technical overview & architecture specification
├── groupguide.md                      # Team orientation, quickstart & viva talking points
├── PROJECT_COMPREHENSIVE_SPECIFICATION.md # Full technical specification & AIAA paper
├── COG_EW_Master_Project_Document.docx# Professional 6-section Word project manual
├── requirements.txt                   # Production dependencies (PyTorch, FastAPI, Gymnasium)
├── dashboard/                         # Mission Control & Presentation Layer
│   ├── index.html                     # Executive Dashboard & System Health overview
│   ├── DOCUMENTATION.html             # HTML export of complete master documentation
│   ├── pages/                         # Core Mission Intelligence Interfaces
│   │   ├── simulation.html            # 3D WebGL Three.js Mission Control & Radar Tactical Screen
│   │   ├── rf_classification.html     # Live 24-class RF modulation oscilloscope & 3D I/Q helix
│   │   ├── signal_analysis.html       # Spectral FFT PSD, constellation & signal quality gauges
│   │   ├── model_performance.html     # Locked test benchmarks & interactive SNR threshold filter
│   │   └── decision_support.html      # Cognitive OODA decision matrix & safety bounds
│   └── assets/                        # CSS styling & procedural Web Audio synthesizers
├── src/
│   └── simulation/                    # Aerospace-Grade Simulation Core
│       ├── atmosphere.py              # US Standard Atmosphere 1976 (0-84 km)
│       ├── aerodynamics.py            # Hypersonic Waverider polars & Sutton-Graves heat flux
│       ├── dynamics.py                # Continuous 14-state space & explicit RK4 50 Hz integrator
│       ├── actuators.py               # Rate-limited elevons (30 deg/s) & cold-gas RCS thrusters
│       ├── ew_physics.py              # Radar range equation, J/S ratio, Barrage & RGPO spoofing
│       ├── threats.py                 # Multi-Static IADS (VHF/S/X-band) & Mach 4 SAM interceptors
│       ├── guidance.py                # Autonomous Guidance Computer ('Mission Brain')
│       └── env.py                     # Gymnasium standard environment (CogEWEnv)
├── perception/                        # RF Perception & Signal Intelligence
│   ├── rf_inference_service.py        # 24-class AMC inference service & DSP analyzer
│   └── models/
│       ├── radioml_resnet.py          # 24-Class Deep ResNet with residual skip connections
│       └── radioml_cnn.py             # Baseline 1D/2D CNN architectures
├── scripts/                           # Core Operational & Benchmark Scripts
│   ├── dashboard_api.py               # Unified FastAPI backend & tactical simulation server
│   ├── benchmark_simulation.py        # 100-seed Monte Carlo evaluation engine
│   ├── test_simulation_flight.py      # End-to-end flight envelope adherence test
│   ├── build_master_word_doc.py       # Professional python-docx master document builder
│   └── export_doc_to_html.py          # Automated HTML documentation synchronizer
└── tests/                             # Comprehensive Automated Test Suite
    ├── test_rk4_simulation.py         # Aerospace unit tests (RK4, aero, atmosphere, RGPO, Gym)
    ├── test_dashboard_api.py          # FastAPI REST endpoints & simulation ticking tests
    └── test_decision_engine.py        # Autonomous decision & EW jamming logic tests
```

---

## ⚡ Quickstart & Mission Control Guide

### 1. Launch Mission Control Backend
```powershell
# In the project root directory:
python -m uvicorn scripts.dashboard_api:app --host 127.0.0.1 --port 8000
```

Open your browser to:
- **3D Tactical Mission Cockpit:** [http://127.0.0.1:8000/pages/simulation.html](http://127.0.0.1:8000/pages/simulation.html)
- **Executive Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **RF 24-Class Modulation Intelligence:** [http://127.0.0.1:8000/pages/rf_classification.html](http://127.0.0.1:8000/pages/rf_classification.html)
- **Spectral FFT & Signal Quality Diagnostics:** [http://127.0.0.1:8000/pages/signal_analysis.html](http://127.0.0.1:8000/pages/signal_analysis.html)
- **Benchmark Leaderboard:** [http://127.0.0.1:8000/pages/model_performance.html](http://127.0.0.1:8000/pages/model_performance.html)
- **Interactive Swagger REST API:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Keyboard Hotkeys (0 ms Latency)
| Key | Function | Description |
|:---:|---|---|
| **Spacebar** | Play / Pause | Instantly toggles simulation ticking with zero round-trip latency |
| **R** | Reset Mission | Re-initializes waverider at $29,500\text{m}$, clears missile debris and tracks |
| **1** | Tactical Orbit | Free 360° mouse drag inspection, zoom, and pitch control |
| **2** | Chase Cam | Dynamic third-person hypersonic pursuit camera behind the Waverider |
| **3** | Cockpit POV | First-person HUD view through the pilot canopy |
| **4** | Wing Cam | Wingtip action camera capturing banking turns and scramjet diamonds |
| **5** | Radar Ground | Ground station perspective looking up from enemy radar site |
| **6** | Target Bunker | Terminal defense view focused on Sector 4 Command Bunker |
| **7** | Top-Down Recon | High-altitude satellite reconnaissance overview |
| **Esc** | Close Dialogs | Dismisses debrief and benchmark modal dialogs |

### 3. Run Automated Verification Tests
```powershell
# 1. Run aerospace unit tests (RK4, Atmosphere, Polars, RGPO, Gymnasium)
python -m pytest tests/test_rk4_simulation.py -v

# 2. Run 100-seed Monte Carlo empirical benchmark
python scripts/benchmark_simulation.py

# 3. Test standoff weapon release & safe egress flight envelope
python scripts/test_simulation_flight.py
```

---

## 📄 License
This project is licensed under the MIT License.
