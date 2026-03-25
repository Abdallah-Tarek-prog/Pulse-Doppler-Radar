import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

c = 3e8
Fc = 76.5e9
lamda = c / Fc
PRI = 5e-6
Tp = 4e-9
Fs = 4e9
startDelayTime = 5e-9
pulsesNum = 1024
SNR = -15
Ts = 1 / Fs
margain = 7

targetsDistance = [45.5, 150.0]
targetsVelocity = [-30, 65]

samplesPerPulse = int(PRI * Fs)
fastTimeArray = np.arange(0, samplesPerPulse) * Ts
timeArray = np.arange(0, pulsesNum * PRI + startDelayTime, Ts)

transmittedSignal = np.zeros_like(timeArray)

for pulseIndex in range(pulsesNum):
    startTime = startDelayTime + pulseIndex * PRI
    startIdx = int(startTime * Fs)
    endIdx = int((startTime + Tp) * Fs)
    if endIdx < len(transmittedSignal):
        transmittedSignal[startIdx:endIdx] = 1

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
            currentPulseSignal[delaySamples:delaySamples + pulseWidthSamples] += 1 * phase

    receivedSignalArray[pulseIndex, :] = currentPulseSignal

noisePower = 10 ** (-SNR / 10)
noiseArray = np.sqrt(noisePower / 2) * (
        np.random.randn(pulsesNum, samplesPerPulse) +
        1j * np.random.randn(pulsesNum, samplesPerPulse)
)
receivedSignalWithNoise = receivedSignalArray + noiseArray

plt.figure(figsize=(10, 4))
plt.plot(timeArray * 1e6, transmittedSignal)
plt.xlabel("Time (us)")
plt.ylabel("Amplitude")
plt.xlim(0, 0.05)
plt.grid()
plt.title("Transmitted Signal (Zoomed)")

min_dist_view = max(0, min(targetsDistance) - margain)
max_dist_view = max(targetsDistance) + margain
minTimeView = ((2 * min_dist_view / c) + startDelayTime) * 1e6
maxTimeView = ((2 * max_dist_view / c) + startDelayTime) * 1e6

plt.figure(figsize=(10, 4))
plt.plot(fastTimeArray * 1e6, np.abs(receivedSignalWithNoise[0]))
plt.xlabel("Fast Time (us)")
plt.ylabel("Amplitude")
plt.xlim(0, 4)
plt.grid()
plt.title(f"Received Signal (First Pulse) @ SNR {SNR}dB")

plt.figure(figsize=(12, 6))

max_range_m = (PRI * c) / 2

plt.imshow(
    np.abs(receivedSignalArray).T,
    aspect="auto",
    extent=[0, pulsesNum, 0, max_range_m],
    cmap="jet",
    origin='lower'
)
plt.xlabel("Slow Time (Pulse Index)")
plt.ylabel("Range (m)")
plt.ylim(min_dist_view, max_dist_view)
plt.colorbar(label="Magnitude")
plt.title("Range-Time Map (Clean Signal - Range in Meters)")
plt.show()

receivedSlice = receivedSignalWithNoise[0, :]

pulseTemplate = np.ones(int(Tp * Fs))
pulseTemplateFlipped = np.flip(pulseTemplate)

startTime = time.time()
rangeCompressed = np.zeros_like(receivedSignalWithNoise)
for n in range(pulsesNum):
    conv_res = np.convolve(receivedSignalWithNoise[n], pulseTemplateFlipped, mode='full')
    rangeCompressed[n, :] = conv_res[len(pulseTemplate) - 1: len(pulseTemplate) - 1 + samplesPerPulse]

matchedFilterOutputTimeDomain = np.sum(np.abs(rangeCompressed), axis=0)
endTime = time.time()
processingTimeMethod1 = (endTime - startTime) * 1000

startTime = time.time()
integrated_fft_acc = np.zeros(samplesPerPulse)
fftSize = 2 ** (int(np.log2(len(receivedSlice))) + 1)
pulseTemplateFft = np.fft.fft(pulseTemplate, fftSize)

for n in range(pulsesNum):
    receivedSliceFft = np.fft.fft(receivedSignalWithNoise[n], fftSize)
    fftProduct = receivedSliceFft * np.conj(pulseTemplateFft)
    fftOutputSingle = np.fft.ifft(fftProduct)
    integrated_fft_acc += np.abs(fftOutputSingle[:samplesPerPulse])

fftOutput = integrated_fft_acc
endTime = time.time()
processingTimeMethod2 = (endTime - startTime) * 1000

peakThreshold = 0.9 * np.max(matchedFilterOutputTimeDomain)

peaksTimeDomainIndices, _ = find_peaks(matchedFilterOutputTimeDomain, height=peakThreshold, distance=int(Tp * Fs) * 2)
peakTimesTimeDomain = fastTimeArray[peaksTimeDomainIndices]
detectedRangesTimeDomain = (peakTimesTimeDomain - startDelayTime) * c / 2

peaksFftIndices, _ = find_peaks(fftOutput, height=peakThreshold, distance=int(Tp * Fs) * 2)
peakTimesFft = fastTimeArray[peaksFftIndices]
detectedRangesFft = (peakTimesFft - startDelayTime) * c / 2

minTimeDomain = minTimeView
maxTimeDomain = maxTimeView
minFft = minTimeView
maxFft = maxTimeView

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(fastTimeArray * 1e6, matchedFilterOutputTimeDomain, color='blue')
plt.title(f'Method 1: Time Domain Matched Filter ')
plt.xlabel('Time (us)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(minTimeDomain, maxTimeDomain)

plt.subplot(4, 1, 2)
plt.plot(fastTimeArray * 1e6, fftOutput, color='green', linestyle='--')
plt.title(f'Method 2: FFT Based Processing ')
plt.xlabel('Time (us)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(minFft, maxFft)

plt.subplot(4, 1, 3)
plt.plot(fastTimeArray * 1e6, matchedFilterOutputTimeDomain, color='blue', label='Signal')
if len(peaksTimeDomainIndices) > 0:
    plt.plot(peakTimesTimeDomain * 1e6, matchedFilterOutputTimeDomain[peaksTimeDomainIndices], marker='o',
             linestyle='none', fillstyle='none', color='red', markersize=12, label='Detected Peaks')
plt.axhline(peakThreshold, color='darkred', linestyle='--', label='Threshold')
plt.title('Peak Detection for Time Domain Method')
plt.xlabel('Time (us)')
plt.legend()
plt.grid(True)
plt.xlim(minTimeDomain, maxTimeDomain)

plt.subplot(4, 1, 4)
plt.plot(fastTimeArray * 1e6, fftOutput, color='green', label='Signal')
if len(peaksFftIndices) > 0:
    plt.plot(peakTimesFft * 1e6, fftOutput[peaksFftIndices], color='red', marker='o', fillstyle='none',
             linestyle='none', markersize=12, label='Detected Peaks')
plt.axhline(peakThreshold, color='darkred', linestyle='--', label='Threshold')
plt.title('Peak Detection for FFT Method')
plt.xlabel('Time (us)')
plt.legend()
plt.grid(True)
plt.xlim(minFft, maxFft)

plt.tight_layout()
plt.show()

print("=" * 60)

dopplerFftSize = pulsesNum
dopplerFrequencyAxis = np.fft.fftshift(np.fft.fftfreq(dopplerFftSize, d=PRI))
velocityAxis = (dopplerFrequencyAxis * c) / (2 * Fc)

detectedVelocitiesList = []
targetMatchIndices = []

plt.figure(figsize=(12, 6))

for detectionIdx, rangeBinIndex in enumerate(peaksTimeDomainIndices):
    slowTimeSignal = rangeCompressed[:, rangeBinIndex]

    dopplerSpectrum = np.fft.fftshift(np.fft.fft(slowTimeSignal, n=dopplerFftSize))
    spectrumMagnitude = np.abs(dopplerSpectrum)

    peakDopplerIndex = np.argmax(spectrumMagnitude)
    estimatedVelocity = velocityAxis[peakDopplerIndex]
    detectedVelocitiesList.append(estimatedVelocity)

    detectedRange = detectedRangesTimeDomain[detectionIdx]
    closestTrueTargetIndex = np.argmin(np.abs(np.array(targetsDistance) - detectedRange))
    targetMatchIndices.append(closestTrueTargetIndex)
    trueVelocity = targetsVelocity[closestTrueTargetIndex]

    movementDirection = "Approaching" if estimatedVelocity > 0 else "Receding"

    plt.subplot(1, len(peaksTimeDomainIndices), detectionIdx + 1)
    plt.plot(velocityAxis, spectrumMagnitude, 'b-', label='Doppler Spectrum')
    plt.axvline(x=estimatedVelocity, color='r', linestyle='--', label=f'Est: {estimatedVelocity:.1f} m/s')
    plt.axvline(x=trueVelocity, color='g', linestyle=':', linewidth=2, label=f'True: {trueVelocity} m/s')

    plt.title(f"Target at ~{detectedRange:.0f}m\n({movementDirection})")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

rangeDopplerMap = np.fft.fft(rangeCompressed, n=pulsesNum, axis=0)
rangeDopplerMapShifted = np.fft.fftshift(rangeDopplerMap, axes=0)
rangeDopplerMapDb = 20 * np.log10(np.abs(rangeDopplerMapShifted) + 1e-10)

rangeAxis = (fastTimeArray - startDelayTime) * c / 2

plt.figure(figsize=(10, 8))
plt.imshow(rangeDopplerMapDb, aspect='auto', extent=[rangeAxis[0], rangeAxis[-1], velocityAxis[0], velocityAxis[-1]],
           origin='lower', cmap='viridis')
plt.colorbar(label='dB')
if len(detectedRangesTimeDomain) > 0:
    plt.plot(detectedRangesTimeDomain, detectedVelocitiesList, 'wx', markersize=10, markeredgewidth=2)
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Range-Doppler Map (SNR {SNR}dB)")
plt.xlim(min_dist_view, max_dist_view)
plt.show()

print(f"Detected Peak Times (Method 1): {peakTimesTimeDomain}")
print(f"Detected Peak Times (Method 2): {peakTimesFft}")

print(f"Detected Ranges (Method 1): {detectedRangesTimeDomain}")
print(f"Detected Ranges (Method 2): {detectedRangesFft}")

print(f"\nProcessing Time (Method 1 - Time Domain): {processingTimeMethod1:.3f} ms")
print(f"Processing Time (Method 2 - FFT Domain): {processingTimeMethod2:.3f} ms")

print("\n--- VELOCITY COMPARISON ---")
print(f"{'Target':<10}{'True Velocity (m/s)':<25}{'Estimated Velocity (m/s)':<30}{'Error (m/s)':<20}{'Direction':<15}")
for i in range(len(detectedVelocitiesList)):
    detectedR = detectedRangesTimeDomain[i]
    closestTrueIdx = np.argmin(np.abs(np.array(targetsDistance) - detectedR))
    rightVelocity = targetsVelocity[closestTrueIdx]

    estVelocity = detectedVelocitiesList[i]
    velocityError = abs(estVelocity - rightVelocity)
    direction = "Approaching" if estVelocity > 0 else "Receding"
    print(f"{i + 1:<10}{rightVelocity:<25.2f}{estVelocity:<30.2f}{velocityError:<20.2f}{direction:<15}")