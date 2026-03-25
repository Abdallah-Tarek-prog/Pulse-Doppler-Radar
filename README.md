# Pulse-Doppler Radar — Simulation & Analysis

**Simulation and Analysis of Target Detection and Velocity Estimation using Pulse-Doppler Radar Principles**

> **Authors:** Abdallah Tarek Sayed Ahmed · Ahmed Essam Afeifi  
> **Course:** Signals & Systems Project  
> **Supervisors:** Dr. Michael Melek & Dr. Mohammed Abdelghany

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Radar System Description](#2-radar-system-description)
3. [System Parameters](#3-system-parameters)
4. [Performance Metrics](#4-performance-metrics)
5. [Signal Processing Pipeline](#5-signal-processing-pipeline)
6. [Simulation Scenarios](#6-simulation-scenarios)
   - [Part 1 — Positive SNR (+10 dB)](#part-1--positive-snr-10-db)
   - [Part 2 — Negative SNR (−15 dB)](#part-2--negative-snr--15-db)
7. [Results & Output Plots](#7-results--output-plots)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [Notes & Corrections to the Report](#9-notes--corrections-to-the-report)
10. [Dependencies & How to Run](#10-dependencies--how-to-run)
11. [References](#11-references)

---

## 1. Project Overview

This project simulates a **Pulse-Doppler Radar** system entirely in Python. Given a set of radar hardware parameters, the code:

- Generates a pulsed transmitted waveform.
- Synthesises realistic received echoes from two moving targets, including **time delay** (range), **Doppler frequency shift** (velocity), and **Additive White Gaussian Noise (AWGN)**.
- Applies two independent matched-filter implementations to compress the range and detect the targets.
- Estimates each target's velocity using a **slow-time FFT (Doppler FFT)** applied along the pulse dimension.
- Produces a **Range-Doppler Map** that simultaneously shows where each target is and how fast it is moving.

Two separate scripts are provided to demonstrate the radar's capability under very different noise conditions:

| Script | SNR | Scenario |
|--------|-----|----------|
| `Code/PositiveSNR.py` | +10 dB | Standard detection — signal is above the noise floor |
| `Code/NegativeSNR.py` | −15 dB | Challenging detection — noise is ~30× stronger than the signal |

---

## 2. Radar System Description

The simulated radar operates at **76.5 GHz** (W-band / mmWave) — a band widely used in automotive and short-range sensing radars. It transmits short rectangular pulses and listens for echoes between pulses.

The key principle exploited is the **Doppler effect**: when a target moves toward or away from the radar, each successive echo arrives with a slightly different phase. By processing the phase variation *across* pulses (slow time), the radar can measure the target's radial velocity with high precision.

The processing chain follows the standard **Pulse-Doppler** architecture:

```
Transmit pulse train
       │
       ▼
Simulate echoes (range delay + Doppler phase shift)
       │
       ▼
Add AWGN noise
       │
       ▼
Matched Filter (Range Compression)   ←── two implementations compared
       │
       ▼
Doppler FFT (Velocity Estimation)
       │
       ▼
Range-Doppler Map + Peak Detection
```

---

## 3. System Parameters

### Part 1 — PositiveSNR.py

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Carrier frequency | Fc | 76.5 GHz |
| Wavelength | λ = c/Fc | ≈ 3.92 mm |
| Pulse Repetition Interval | PRI | 4 µs |
| Pulse Repetition Frequency | PRF = 1/PRI | 250 kHz |
| Pulse width | Tp | 5 ns |
| Sampling frequency | Fs | 2 GHz |
| Number of pulses | N | 256 |
| Signal-to-Noise Ratio | SNR | +10 dB |

### Part 2 — NegativeSNR.py

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Carrier frequency | Fc | 76.5 GHz |
| Wavelength | λ = c/Fc | ≈ 3.92 mm |
| Pulse Repetition Interval | PRI | 5 µs |
| Pulse Repetition Frequency | PRF = 1/PRI | 200 kHz |
| Pulse width | Tp | 4 ns |
| Sampling frequency | Fs | 4 GHz |
| Number of pulses | N | 1024 |
| Signal-to-Noise Ratio | SNR | −15 dB |

---

## 4. Performance Metrics

### Part 1 (PRI = 4 µs, Tp = 5 ns)

| Metric | Formula | Value |
|--------|---------|-------|
| Maximum unambiguous range | R_max = c × PRI / 2 | **600 m** |
| Maximum unambiguous velocity | v_max = c × PRF / (4 × Fc) | **245 m/s** |
| Minimum detectable range (blind zone) | R_min = c × Tp / 2 | **0.75 m** |

### Part 2 (PRI = 5 µs, Tp = 4 ns)

| Metric | Formula | Value |
|--------|---------|-------|
| Maximum unambiguous range | R_max = c × PRI / 2 | **750 m** |
| Maximum unambiguous velocity | v_max = c × PRF / (4 × Fc) | **196 m/s** |
| Minimum detectable range (blind zone) | R_min = c × Tp / 2 | **0.6 m** |

---

## 5. Signal Processing Pipeline

### 5.1 Transmitted Signal

A rectangular pulse of width `Tp` is generated at the start of each PRI, offset by a small `startDelayTime` guard interval. The full pulse train spans `N × PRI` seconds.

### 5.2 Received Signal Synthesis

For each target `i` with range `r_i` and radial velocity `v_i`:

- **Round-trip time delay:**  
  `τ_i = 2 × r_i / c + startDelayTime`

- **Doppler frequency:**  
  `f_d = 2 × v_i / λ`

- **Complex phase for pulse `n`:**  
  `φ[n] = exp(j × 2π × f_d × (n × PRI + startDelayTime))`

The echo is a copy of the transmitted pulse, placed at sample `τ_i × Fs`, multiplied by the Doppler phase for that pulse. Echoes from all targets are summed.

### 5.3 Noise Model

Complex AWGN is added:

```
noise_power = 10^(−SNR/10)
noise = sqrt(noise_power/2) × (randn + j×randn)
```

### 5.4 Matched Filter (Range Compression)

Two mathematically equivalent implementations are compared:

**Method 1 — Time-Domain Convolution**  
Each pulse row is convolved with the flipped pulse template. The output for all pulses is incoherently summed (magnitude integration). Complexity: O(N²) per pulse.

**Method 2 — FFT-Domain Multiplication**  
The matched filter is implemented as element-wise multiplication of the received signal FFT and the conjugate of the template FFT, followed by an IFFT. Complexity: O(N log N) per pulse.

Both methods produce identical detection results, confirming that convolution in the time domain equals multiplication in the frequency domain.

**Processing time comparison (Part 1, N=256):**

| Method | Processing Time |
|--------|----------------|
| Time-Domain Matched Filter | ~0.77 ms |
| FFT-Based | ~3.59 ms |

> For small datasets (256 pulses), time-domain convolution is faster due to low FFT overhead. FFT becomes significantly faster as dataset size or signal frequency grows.

### 5.5 Range Detection & Peak Picking

After integration:
- A threshold is set as a fraction of the peak magnitude.
- `scipy.signal.find_peaks` locates peaks above the threshold.
- Detected range is computed as:  
  `R = c × (t_peak − startDelayTime) / 2`

### 5.6 Velocity Estimation (Doppler FFT)

For each detected range bin, the **slow-time signal** (one sample per pulse) is extracted from the range-compressed data matrix. An FFT over this slow-time vector yields the Doppler spectrum. The peak Doppler bin maps to:

`v_estimated = f_doppler × c / (2 × Fc)`

**Sign convention:** Positive velocity = target moving toward the radar (approaching); Negative velocity = target moving away (receding).

### 5.7 Range-Doppler Map

A 2-D FFT is applied along the slow-time (pulse) axis of the range-compressed matrix. The result, displayed in dB, gives a bird's-eye view of all targets simultaneously in range-velocity space.

---

## 6. Simulation Scenarios

### Part 1 — Positive SNR (+10 dB)

**Target scenario:**

| Target | True Range (m) | True Velocity (m/s) | Direction |
|--------|---------------|---------------------|-----------|
| A | 150 | −12.56 | Receding |
| B | 45.5 | +50 | Approaching |

**Detection results:**

| Target | True Range | Detected Range (MF) | Detected Range (FFT) |
|--------|-----------|---------------------|----------------------|
| A | 150 m | 149.925 m | 149.925 m |
| B | 45.5 m | 45.45 m | 45.45 m |

**Velocity estimation:**

| Target | True Velocity | Estimated Velocity | Error | Direction |
|--------|--------------|-------------------|-------|-----------|
| A | −12.56 m/s | −13.4 m/s | 0.84 m/s | Receding |
| B | +50 m/s | +49.79 m/s | 0.21 m/s | Approaching |

**Why SNR = +10 dB?**  
At 10 dB the received signal is visibly noisy in the time domain — a simple threshold would miss targets — but matched filtering cleanly recovers both. This is the ideal demonstration point: difficult enough to require signal processing, but straightforward enough to validate results.

---

### Part 2 — Negative SNR (−15 dB)

**Target scenario:**

| Target | True Range (m) | True Velocity (m/s) | Direction |
|--------|---------------|---------------------|-----------|
| A | 45.5 | −30 | Receding |
| B | 150 | +65 | Approaching |

**Why SNR = −15 dB?**  
At −15 dB the noise power is approximately 30× stronger than the signal power. In the raw time-domain plot, the signal is completely invisible — it looks like pure noise. This scenario demonstrates the **processing gain** achievable by combining matched filtering with coherent pulse integration across 1024 pulses. The gain scales as `10 × log10(N)` ≈ 30 dB, which is just enough to push both targets above the noise floor.

Chosen rationale:
- **−20 dB** — too extreme; even with 1024 pulses the targets are unreliably detected (frequent false alarms).
- **−5 dB** — too easy; targets emerge without needing the full integration chain.
- **−15 dB** — sweet spot: requires the entire processing chain but produces clean, repeatable results.

---

## 7. Results & Output Plots

Each script generates the following sequence of figures:

| # | Figure | Description |
|---|--------|-------------|
| 1 | Transmitted Signal | Pulse train showing the rectangular waveform over time |
| 2 | Received Signal (1st pulse) | Single received pulse, clearly noise-dominated at low SNR |
| 3 | Range-Time Map (raw) | 2D image: all pulses (slow time) vs. sample index (fast time) before processing |
| 4 | Range-Time Map (matched filter output) | Same map after range compression — targets appear as bright horizontal streaks |
| 5 | Integrated Matched Filter output — both methods | Overlay of time-domain and FFT-domain integrated signals |
| 6 | Peak detection — Method 1 | Integrated signal with detected peaks and threshold line |
| 7 | Peak detection — Method 2 | Same for the FFT method |
| 8 | Doppler Spectrum (per target) | Slow-time FFT for each detected range bin, showing estimated vs. true velocity |
| 9 | Range-Doppler Map | 2D map in dB; white × markers indicate detected target positions |

---

## 8. Challenges & Solutions

### Challenge 1 — Correct Delay Alignment

**Problem:** The radar cannot transmit and receive simultaneously. A `startDelayTime` guard interval is needed, and all range calculations must subtract this offset or targets appear at the wrong distance.

**Solution:** The round-trip delay is defined as `τ = 2R/c + startDelayTime`. When computing detected range from a peak time `t_peak`, the offset is subtracted back:  
`R = c × (t_peak − startDelayTime) / 2`

---

### Challenge 2 — Phase-Coherent Doppler Modelling

**Problem:** Simply delaying the pulse is not enough — the Doppler effect causes each successive pulse to arrive with a slightly different phase. Without this, the Doppler FFT would see no frequency shift.

**Solution:** Each echo is multiplied by `exp(j × 2π × f_d × (n × PRI + startDelayTime))`, where `n` is the pulse index. This explicitly encodes the inter-pulse phase progression that the Doppler FFT later measures.

---

### Challenge 3 — Target Visibility at Negative SNR

**Problem:** At SNR = −15 dB the received signal is buried under noise. A single pulse is completely undetectable.

**Solution:** Two complementary processing gains are stacked:
1. **Pulse Compression (Matched Filter)** — correlating the received signal with the pulse template sharpens the target peak in range by a factor of `Tp × Fs` (time-bandwidth product).
2. **Coherent Pulse Integration** — processing 1024 pulses and integrating the matched filter outputs adds `~10 × log10(1024) ≈ 30 dB` of SNR gain, bringing buried targets above the detection threshold.

---

### Challenge 4 — Peak Detection Robustness

**Problem:** After integration, sidelobes and noise ripple can cause spurious detections or missed peaks if the threshold is too low or too high.

**Solution:**
- For the positive-SNR scenario (Part 1), the threshold is set at **50%** of the maximum peak (`0.5 × max`), which comfortably rejects sidelobes while capturing both targets.
- For the negative-SNR scenario (Part 2), the threshold is raised to **90%** (`0.9 × max`) and a minimum inter-peak distance is enforced (`distance = 2 × Tp × Fs`) to prevent the matched filter sidelobes from being counted as separate targets.

---

### Challenge 5 — FFT Overhead vs. Time-Domain Speed

**Problem:** For small pulse counts, the FFT-based matched filter is actually *slower* than the direct convolution, contradicting the common expectation that "FFT is always faster".

**Explanation:** The FFT implementation here zero-pads each pulse to the next power of two, computes forward FFT, multiplies, then computes inverse FFT — three FFT operations per pulse. For short signals (Part 1: 8000 samples, 256 pulses) the bookkeeping overhead exceeds the computational savings. For the larger dataset (Part 2: 20000 samples, 1024 pulses) the O(N log N) advantage becomes clear.

---

### Challenge 6 — Velocity Quantisation Error

**Problem:** The Doppler FFT has a finite frequency resolution of `1 / (N × PRI)`. The true Doppler frequency may fall between two FFT bins, causing a small bias in the estimated velocity.

**Solution:** The resolution improves with a larger number of pulses. Part 2 uses N = 1024 instead of N = 256, giving approximately 4× finer Doppler resolution and smaller velocity errors.

---

## 9. Notes & Corrections to the Report

The accompanying PDF report (`Abdallah&Ahmed(SignalsProject).pdf`) contains several inaccuracies identified during review:

| # | Location in PDF | Statement in PDF | Correct Value / Explanation |
|---|----------------|------------------|-----------------------------|
| 1 | Introduction | "X-band (8–12 GHz)" | The carrier frequency is **76.5 GHz**, which places this radar in the **W-band (75–110 GHz)**, not X-band. W-band is the standard for automotive and short-range mmWave radars. |
| 2 | Part 1, Section 3.3 | "the radar cannot detect any target closer than **75 meters**" | The formula gives R_min = c × Tp / 2 = (3×10⁸ × 5×10⁻⁹) / 2 = **0.75 m**. The text erroneously inflates this by a factor of 100. |
| 3 | Part 2, Section 3.3 | "0.75 meters (as stated in the description)" | For Part 2 the formula gives R_min = (3×10⁸ × 4×10⁻⁹) / 2 = **0.6 m**, not 0.75 m. The two scenarios have different pulse widths and therefore different blind-zone distances. |
| 4 | Part 1, Analysis Conclusion | "FFT is generally faster" (implying always faster) | For the small dataset in Part 1 (N=256), **time-domain convolution is actually faster** (~0.77 ms vs. ~3.59 ms). FFT only outperforms direct convolution at larger data sizes. The code measurements confirm this. |

---

## 10. Dependencies & How to Run

### Requirements

```
Python ≥ 3.8
numpy
matplotlib
scipy
```

Install with:

```bash
pip install numpy matplotlib scipy
```

### Run the positive-SNR scenario

```bash
python Code/PositiveSNR.py
```

### Run the negative-SNR scenario

```bash
python Code/NegativeSNR.py
```

Each script runs end-to-end and displays all output plots sequentially. No input files are required.

---

## 11. References

**Python Libraries**
- NumPy — https://numpy.org/
- Matplotlib — https://matplotlib.org/stable/

**Learning Resources**
- Radar fundamentals (YouTube) — https://youtu.be/NtyU6aKZ-cY
- Pulse-Doppler tutorial (YouTube) — https://youtu.be/UO98lJQ3QGI
- Range-Doppler processing (YouTube) — https://www.youtube.com/watch?v=OnDIGIvTGcc
- Matched filter walkthrough (YouTube) — https://youtu.be/VXU4LSAQDSc
- Doppler radar overview — https://radarsimx.com/2019/05/16/doppler-radar/
- Introduction to Radar Part 2 — https://medium.com/@itberrios6/introduction-to-radar-part-2-8a332066917e
- MATLAB Range-Doppler Response — https://www.mathworks.com/help/phased/ug/range-doppler-response.html
