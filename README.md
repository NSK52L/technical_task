# Sustainable Orchestration – Technical Assessment

This repository contains a small simulation developed for the internship technical assessment. It explores energy-aware orchestration in a simplified edge–cloud computer-vision scenario.

The simulation uses synthetic data and estimated device/model behavior. It is not a hardware implementation or a validated energy measurement.

## 1. Scenario and topology

The simulated system represents a camera-based monitoring application with:

- **Camera:** periodically captures images.
- **Edge computer:** a small computer that can execute a computer-vision model.
- **Cloud computer:** a more powerful computer that can also execute processing workloads.
- **Computer-vision processing:** modeled detection workload, with normal and throttled operating modes.
- **Lamp:** optional load that can be switched on or off.
- **Storage and cooling:** storage temperature is simulated, and a fan can be activated according to the temperature model.
- **Energy/environment model:** device power and environmental conditions are represented by synthetic models.

The components and their interactions are represented in software. Physical devices, interfaces, and real power meters were not connected during this assessment.

## 2. Consumption modeling and adaptation strategies

Energy consumption is calculated from the modeled power of the system components over the simulation time steps. The models and training data are synthetic; therefore, the results should be interpreted as simulation estimates rather than measured consumption.

The simulation considers configuration changes such as:

- Switching the lamp on or off.
- Selecting normal or throttled processing.
- Choosing processing configurations evaluated by the orchestrator.

The trained models estimate the behavior of candidate configurations. The orchestrator uses these estimates when comparing configurations.

## 3. Orchestrator and requirements

The orchestrator evaluates available configurations and selects a configuration intended to reduce energy consumption while considering system requirements and budgets.

The simulated requirements include:

- **Detection requirement:** at least one successful detection in each rolling 60-second window.
- **Power budget:** total power must not exceed 1 kW.
- **Storage temperature:** must remain below 50 °C.

The simulation records system behavior, energy consumption, detections, and configuration changes.

The current implementation demonstrates model-based configuration selection. It does not implement a complete forecasting-and-planning system or a formally verified constraint solver.

## 4. Configuration feasibility

The candidate evaluation considers predicted system behavior and the stated requirements. However, the current implementation does not provide a formal guarantee that it can identify every infeasible configuration space.

The results also show that an energy-saving configuration can reduce performance and may fail the detection requirement. This is an important limitation of the current approach and indicates that energy minimization must be balanced against reliability requirements.

## 5. Simulation setup

- Simulation duration: **24 hours**
- Baseline and adaptive runs use the same generated environmental trace for comparison.
- Results are produced using synthetic device behavior and trained models.
- The simulation is software-based; deployment in a container or virtual edge–cloud environment was not completed.

## 6. Results

| Metric | Fixed baseline | Adaptive |
|---|---:|---:|
| Simulated energy | 1.30260 kWh | 1.01544 kWh |
| Energy reduction | — | **22.05%** |
| Captured images | 7,200 | 2,880 |
| Processed images | 7,200 | 2,880 |
| Successful detections | 6,018 | 1,985 |
| Maximum total power | 54.305 W | 42.430 W |
| Power-budget violation steps | 0 | 0 |
| Failed rolling 60-second detection windows | 3 | 217 |
| Final storage temperature | 35.53 °C | 35.24 °C |
| Lamp-on steps | 8,640 | 0 |
| Configuration changes | 0 | 1 |

### Interpretation

The adaptive run reduced simulated energy consumption by **22.05%**, from 1.30260 kWh to 1.01544 kWh. Its maximum modeled power was also lower than the baseline. Neither run exceeded the 1 kW power budget, and both remained below the 50 °C storage-temperature limit.

However, the adaptive run captured and processed fewer images and had substantially more failed rolling detection windows (217 compared with 3 in the baseline). Therefore, the energy reduction came with a significant loss in detection-service performance. The current adaptive strategy does **not** satisfy the detection requirement reliably.

These results demonstrate the trade-off between energy consumption and service quality in this simplified scenario. They should not be interpreted as proof that the requirements are simultaneously satisfied or that the models reflect real hardware.

## 7. Limitations and next steps

- Device power and behavior are synthetic estimates, not real measurements.
- No physical camera, computers, lamp, smart plugs, or power meters were integrated.
- No container or virtual-machine deployment was completed.
- The simulation does not provide formal infeasibility detection for all possible configurations.
- The adaptive strategy does not reliably satisfy the rolling detection requirement in the final run.
- The results depend on the synthetic models, generated data, and simulation assumptions.

A next step would be to connect the simulation to measurements from the intended devices, calibrate the models, and improve the orchestrator's handling of service constraints before evaluating it on a real edge–cloud setup.

## 8. Reproducing the simulation

Create and activate a Python virtual environment, install the dependencies, and run the final adaptive simulation script.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python adaptive_simulation.py
```

The script generates the comparison and simulation result files. Use the exact scripts and model artifacts included in this repository.

## 9. Generative AI declaration

Generative AI was used as an assistance tool during development for explanations, debugging support, code review, and documentation drafting. The simulation assumptions, implementation, execution, and interpretation of the results were reviewed by me.
