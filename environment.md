# Hardware and Software Environment

## Hardware

- Platform: NVIDIA Jetson AGX Xavier
- Deployment setting: embedded edge-device evaluation

## Software

- Operating system: Ubuntu / Jetson Linux
- Programming language: Python 3
- Deep learning framework: PyTorch
- Monitoring tool: tegrastats where available

## Notes

Direct board-level power measurement was not available from the exposed tegrastats output in the current software environment. Therefore, energy is discussed only as a coarse estimate based on runtime and assumed average device power. The main system-level evidence is based on communication cost, runtime, GPU utilization, CPU utilization, and memory behavior where available.
