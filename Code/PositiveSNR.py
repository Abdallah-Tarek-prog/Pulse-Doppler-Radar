import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

c = 3e8
Fc = 76.5e9
lamda = c / Fc
PRI = 4e-6
Tp = 5e-9
Fs = 2e9
startDelayTime = 5e-9
pulsesNum = 256
SNR = 10
Ts = 1 / Fs
margain = 2

samplesPerPulse = int(PRI * Fs)
fastTimeArray = np.arange(0, samplesPerPulse) * Ts

targetsDistance = [150, 45.5]
targetsVelocity = [-12.56, 50]

timeArray = np.arange(0, pulsesNum * PRI + startDelayTime, Ts)
transmittedSignal = np.zeros_like(timeArray)

for pulseIndex in range(pulsesNum):
    startTime = startDelayTime + pulseIndex * PRI
    transmittedSignal[(timeArray >= startTime) & (timeArray < startTime + Tp)] = 1

receivedSignalArray = np.zeros((pulsesNum, samplesPerPulse), dtype=complex)

for pulseIndex in range(pulsesNum):
    currentPulseSignal = np.zeros(samplesPerPulse, dtype=complex)

    for i in range(len(targetsDistance)):
        r = targetsDistance[i]
        v = targetsVelocity[i]
        fd = (2 * v) / lamda

        tau = (2 * r / c) + startDelayTime
        delaySamples = int(tau * Fs)
        pulseWidthSamples = int(Tp * Fs)

        if delaySamples + pulseWidthSamples < samplesPerPulse:
            phase = np.exp(1j * 2 * np.pi * fd * (pulseIndex * PRI + startDelayTime))
            currentPulseSignal[delaySamples: delaySamples + pulseWidthSamples] += 1 * phase

    receivedSignalArray[pulseIndex, :] = currentPulseSignal

noisePower = 10 ** (-SNR / 10)
noise = np.sqrt(noisePower / 2) * (
        np.random.randn(pulsesNum, samplesPerPulse) +
        1j * np.random.randn(pulsesNum, samplesPerPulse)
)

receivedSignalWithNoise = receivedSignalArray + noise

plt.figure(figsize=(10, 4))
plt.plot(timeArray * 1e6, transmittedSignal)
plt.xlabel("Time (us)")
plt.ylabel("Amplitude")
plt.xlim(0, 30)
plt.grid()
plt.title("Transmitted Signal")

plt.figure(figsize=(10, 4))
plt.plot((fastTimeArray * 1e6), np.abs(receivedSignalWithNoise[0]))
plt.xlabel("Fast Time (us)")
plt.ylabel("Amplitude")
plt.xlim(0, PRI * 1e6)
plt.grid()
plt.title(f"Received Signal (First Pulse) @ SNR {SNR}dB")

plt.figure(figsize=(8, 6))
plt.imshow(
    np.abs(receivedSignalWithNoise).T,
    aspect="auto",
    extent=[0, pulsesNum, 0, PRI / Ts],
    cmap="jet",
    origin='lower'
)
plt.xlabel("Slow Time (Pulse Index)")
plt.ylabel("Range (us)")
plt.colorbar(label="Magnitude")
plt.title("Range-Time Map (Raw Signal)")
plt.show()

pulseTemplate = np.ones(int(Tp * Fs))
pulseTemplateFlipped = np.flip(pulseTemplate)

rangeCompressed = np.zeros_like(receivedSignalWithNoise)

for n in range(pulsesNum):
    convResult = np.convolve(receivedSignalWithNoise[n], pulseTemplateFlipped, mode='full')
    rangeCompressed[n, :] = convResult[len(pulseTemplate) - 1: len(pulseTemplate) - 1 + samplesPerPulse]

t0 = time.time()
integratedSignalTime = np.sum(np.abs(rangeCompressed), axis=0)
t1 = time.time()
timeMF = (t1 - t0) * 1000

plt.figure(figsize=(8, 6))
plt.imshow(
    np.abs(rangeCompressed).T,
    aspect="auto",
    extent=[0, pulsesNum, 0, PRI * 1e6],
    cmap="jet",
    origin='lower'
)
plt.xlabel("Slow Time (Polse Index)")
plt.ylabel("Fast Time / Range (us)")
plt.colorbar(label="Magnitude")
plt.title(f"Range-Time Map (Matched Filter Output)")
plt.show()

startTimeFFT = time.time()
integratedSignalFFT = np.zeros(samplesPerPulse)
nFFT = 2 ** (int(np.log2(samplesPerPulse)) + 1)
pulseTemplateFFT = np.fft.fft(pulseTemplate, nFFT)

for pulseIdx in range(pulsesNum):
    receivedFFT = np.fft.fft(receivedSignalWithNoise[pulseIdx, :], nFFT)
    matchedFilterFFT = receivedFFT * np.conj(pulseTemplateFFT)
    matchedFilterOutput = np.fft.ifft(matchedFilterFFT)
    integratedSignalFFT += np.abs(matchedFilterOutput[:samplesPerPulse])

endTimeFFT = time.time()
processingTimeFFT = (endTimeFFT - startTimeFFT) * 1000

peakThreshold = 0.5 * np.max(integratedSignalTime)
peaksIdxMF, _ = find_peaks(integratedSignalTime, height=peakThreshold, distance=int(Tp * Fs))
peakTimesMF = fastTimeArray[peaksIdxMF]
detectedRangesMF = (peakTimesMF - startDelayTime) * c / 2

peaksIdxFFT, _ = find_peaks(integratedSignalFFT, height=peakThreshold, distance=int(Tp * Fs))
peakTimesFFT = fastTimeArray[peaksIdxFFT]
detectedRangesFFT = (peakTimesFFT - startDelayTime) * c / 2

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot((fastTimeArray * 1e6) + (startDelayTime * 1e6), integratedSignalTime, 'b', label='Integrated')
plt.title(f"Method 1: Integrated Matched Filter Time Domain ")
plt.xlabel("Time (us)")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot((fastTimeArray * 1e6) + (startDelayTime * 1e6), integratedSignalFFT, 'g--', label='Integrated')
plt.title(f"Method 2: FFT ")
plt.xlabel("Time (us)")
plt.ylabel("Magnitude")
plt.grid()

plt.tight_layout()
plt.show()

if len(peaksIdxMF) > 0:
    minTimeDomain = peakTimesMF.min() * 1e6
    maxTimeDomain = peakTimesMF.max() * 1e6

    plt.figure(figsize=(10, 6))
    plt.plot(fastTimeArray * 1e6, integratedSignalTime, color='blue', label='Signal')
    plt.plot(peakTimesMF * 1e6, integratedSignalTime[peaksIdxMF], marker='o', linestyle='none', fillstyle='none',
             color='red', markersize=12, label='Detected Peaks')
    plt.axhline(peakThreshold, color='darkred', linestyle='--', label='Threshold')
    plt.title('Peak Detection for Time Domain Method')
    plt.xlabel('Time (us)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.xlim(0 , maxTimeDomain + margain)
    plt.show()

if len(peaksIdxFFT) > 0:
    minFft = peakTimesFFT.min() * 1e6
    maxFft = peakTimesFFT.max() * 1e6

    plt.figure(figsize=(10, 6))
    plt.plot(fastTimeArray * 1e6, integratedSignalFFT, color='green', label='Signal')
    plt.plot(peakTimesFFT * 1e6, integratedSignalFFT[peaksIdxFFT], color='red', marker='o', fillstyle='none',
             linestyle='none', markersize=12, label='Detected Peaks')
    plt.axhline(peakThreshold, color='darkred', linestyle='--', label='Threshold')
    plt.title('Peak Detection for FFT Method')
    plt.xlabel('Time (us)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, maxFft + margain)
    plt.show()

N_Doppler = pulsesNum
dopplerFreq = np.fft.fftshift(np.fft.fftfreq(N_Doppler, d=PRI))
velocityAxis = (dopplerFreq * c) / (2 * Fc)

detectedVelocities = []

for i, rangeIdx in enumerate(peaksIdxMF):
    slowTimeSignal = rangeCompressed[:, rangeIdx]

    dopplerSpectrum = np.fft.fftshift(np.fft.fft(slowTimeSignal, n=N_Doppler))
    magnitudeSpectrum = np.abs(dopplerSpectrum)

    peakDopplerIdx = np.argmax(magnitudeSpectrum)
    estVelocity = velocityAxis[peakDopplerIdx]
    detectedVelocities.append(estVelocity)

    detectedR = detectedRangesMF[i]
    closestTrueIdx = np.argmin(np.abs(np.array(targetsDistance) - detectedR))
    trueV = targetsVelocity[closestTrueIdx]

    plt.figure(figsize=(6, 4))
    plt.plot(velocityAxis, magnitudeSpectrum)
    plt.axvline(estVelocity, color='r', linestyle='--', label=f'Est: {estVelocity:.2f}')
    plt.axvline(trueV, color='g', linestyle=':', label=f'True: {trueV}')
    plt.title(f"Target at {detectedR:.0f}m")
    plt.xlabel("Velocity (m/s)")
    plt.legend()
    plt.grid()
    plt.show()

rangeDopplerMap = np.fft.fft(rangeCompressed, n=N_Doppler, axis=0)
rangeDopplerMapShifted = np.fft.fftshift(rangeDopplerMap, axes=0)
rangeDopplerMapDb = 20 * np.log10(np.abs(rangeDopplerMapShifted) + 1e-10)

rangeAxis = (fastTimeArray - startDelayTime) * c / 2

plt.figure(figsize=(10, 8))
plt.imshow(rangeDopplerMapDb, aspect='auto', extent=[rangeAxis[0], rangeAxis[-1], velocityAxis[0], velocityAxis[-1]],
           origin='lower', cmap='jet')
plt.colorbar(label='dB')
if len(detectedRangesMF) > 0:
    plt.plot(detectedRangesMF, detectedVelocities, 'wx', markersize=10, markeredgewidth=2)
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Range-Doppler Map (SNR {SNR}dB)")
plt.show()

print(f"Detected Peak Times (Method 1): {peakTimesMF}")
print(f"Detected Peak Times (Method 2): {peakTimesFFT}")

print(f"Detected Ranges (Method 1): {detectedRangesMF}")
print(f"Detected Ranges (Method 2): {detectedRangesFFT}")

print(f"\nProcessing Time (Method 1 - Time Domain): {timeMF:.3f} ms")
print(f"Processing Time (Method 2 - FFT Domain): {processingTimeFFT:.3f} ms")

print("\n--- VELOCITY COMPARISON ---")
print(f"{'Target':<10}{'True Velocity (m/s)':<25}{'Estimated Velocity (m/s)':<30}{'Error (m/s)':<20}{'Direction':<15}")
for i in range(len(detectedVelocities)):
    rightVelocity = detectedVelocities[i] if i < len(detectedVelocities) else None
    estVelocity = detectedVelocities[i]
    velocityError = estVelocity - rightVelocity
    direction = "Approaching" if estVelocity > 0 else "Receding"
    print(f"{i + 1:<10}{rightVelocity:<25.2f}{estVelocity:<30.2f}{velocityError:<20.2f}{direction:<15}")