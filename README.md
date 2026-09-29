## Trajectory Tracking Controller

This hardware-in-the-loop (HIL) tracking architecture is designed for differential-drive swarm robots. By bridging a Python and OpenCV vision pipeline with a dual-core ESP32 hardware node via UDP, the system executes precise parametric trajectories and corrects spatial errors in real-time.

## Project Status
**Active / Work in Progress.** We are currently validating the baseline differential-drive kinematics on a single agent. Future updates will scale this architecture to support multi-robot flocking and complex lattice formations.

## Repository Structure

*   **Trajectory Generation (`dummy_trajectory/`)**: Mathematical trajectory scripts tailored to physical arena bounds, outputting hardware-ready CSV files.
*   **Vision Pipeline (`vision_pipeline/`)**: Converts camera frames to grayscale, extracting ArUco marker poses using relaxed polygon approximations, Exponential Moving Average (EMA) filtering, and homography mapping.
*   **Kinematics Engine (`kinematic_controller/`)**: Translates spatial and heading errors into independent wheel velocities. It utilizes proportional deadband scaling to overcome static floor friction without breaking the unicycle turning radius.
*   **Master Dispatcher (`swarm_controller.py`)**: Loops over the trajectory CSV, applying acceleration ramping (slew rate limiting) and optical dead reckoning to command the hardware safely.
*   **Hardware Firmware (`firmware/`)**: A FreeRTOS C++ application where Core 0 handles asynchronous UDP packets and Core 1 runs a strict 50ms feedforward-assisted positional PID control loop.

## Quick Start Guide

*   Execute `generate_s_curve.py` to calculate the baseline mathematical path.
*   Flash the ESP32 firmware sketch to the microcontroller using the Arduino IDE.
*   Run `swarm_controller.py` on the central PC.
*   Wait for the camera feed to stabilize, then press `c` to lock the arena calibration and initiate tracking.
