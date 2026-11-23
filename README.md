# Drone-Irrigation-System-CSE299-Project
```markdown
# Camera-Guided Zone Irrigation for Small Plots

A low-cost, closed-loop irrigation system for small plots and four-zone test rigs that waters only dry zones instead of running a uniform, fixed schedule. Built for **CSE 299** (Fall 2026) at **North South University**.

---

## Overview

Uniform irrigation wastes water and pumping energy while leaving dry patches stressed. Existing aerial-sensing approaches rely on expensive thermal or multispectral cameras and stop at generating stress maps rather than controlling irrigation hardware. This project closes the loop within a **BDT 8,000** hardware budget by combining an overhead **ESP32-CAM** RGB survey with **capacitive soil-moisture probes**, **mini DC pumps**, an **ESP32 DevKit**, and a local **Python / OpenCV** pipeline.

---

## System Architecture

```text
[ESP32-CAM on F450 Frame] ──(JPEG / HTTP)──▶ [Laptop: Python + OpenCV VARI Map & Decision Engine]
                                                               │
                                                     (HTTP Commands / Readings)
                                                               ▼
[4 Capacitive Soil Probes] ──(ADC1)──▶ [ESP32 DevKit + 4-Ch Relay] ──▶ [4 Mini DC Pumps]
                                           └─ Local soil-probe fallback on Wi-Fi/laptop loss

```

* **Vision & Stress Index:** Computes the Visible Atmospherically Resistant Index ($\text{VARI} = \frac{G - R}{G + R - B}$) per zone on the laptop to classify low, medium, or high crop stress.
* **Sensor Fusion & Control:** Combines image stress classes with capacitive soil-moisture readings (connected to ESP32 `ADC1` pins) to trigger timed watering pulses (`20 s` pulse, `60 s` soak interval) only in dry zones.
* **Offline Fail-Safe:** Operates entirely on a local network without cloud dependencies; if Wi-Fi or the laptop drops, the ESP32 falls back to local soil-probe thresholds and enforces pump runtime limits.

---

## Hardware & Software Stack

| Component | Details |
| --- | --- |
| **Camera Node** | ESP32-CAM (OV2640) mounted on an F450 frame for overhead survey framing |
| **Controller Node** | ESP32 DevKit (Wi-Fi + `ADC1` analog inputs for soil sensors) |
| **Sensors & Actuators** | 4× Capacitive soil-moisture probes, 4-channel relay module, 4× Mini DC pumps, 12V 2A supply |
| **Processing & UI** | Python 3, OpenCV, NumPy, local HTML/CSV logging on laptop |

---

## Performance Targets

* **Zone Stress Agreement:** $\ge 80\%$ agreement between camera zone classification and soil-probe ground truth
* **Latency:** $\le 5\text{ minutes}$ from survey capture to first pump command
* **Water Savings:** $\ge 30\%$ reduction in water used compared to a fixed irrigation schedule
* **Fault Safety:** Pump stops within `10 s` on fault; maximum `60 s` per pulse
* **Total Cost:** $\le \text{BDT } 8,000$ (~BDT 6,750 estimated build cost)

---

## Repository Structure

```text
├── camera_analysis/       # OpenCV scripts for ROI cropping, VARI mapping, and health classification
├── esp32_firmware/        # ESP32-CAM capture & ESP32 DevKit soil-probe / relay pump control firmware
├── dashboard/             # Local web dashboard and CSV data logger
└── docs/                  # Proposal slides, wiring diagrams, and test logs

```

---

## Quick Start

1. **Install Python dependencies:**
```bash
pip install opencv-python numpy

```


2. **Power the ESP32-CAM and ESP32 DevKit** and connect your laptop to the local Wi-Fi network (`192.168.4.1`).
3. **Run the capture & analysis script:**
```bash
python camera_analysis/vari_stream.py

```


4. **Adjust ROI & thresholds** in the OpenCV controller window, then press `c` or `Enter` to capture and analyze a frame (`q` to quit).

---

## Team & Supervision

* **Deep Barua** (`2231332642`) — Team Lead (Drone & Imaging)
* **Salim Shadman Soron** (`2514545642`) — Embedded, Docs & QA
* **Mirza Zareef Ahmed** (`2212680042`) — Technical Lead (Decision Logic & Dashboard)
* **Supervisor:** Mirza Mohammad Lutfe Elahi, Department of Electrical and Computer Engineering, North South University



```

```
