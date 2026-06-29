# Physics model

This document describes the forward and inverse model behind the simulator. All of it is
analytic (no Monte Carlo / FEM), chosen so the app stays a single, fast, inspectable file
while remaining physically grounded.

## 1. Cap geometry

A real high-density montage: **26 sources, 31 detectors** on an interleaved planar lattice
(first-nearest-neighbour separation 10 mm). Every source-detector pair within 40 mm is a
measurement, which yields discrete nearest-neighbour separation bands:

| Band | Separation | Role |
|---|---|---|
| NN1 | ~10 mm | scalp / superficial |
| NN2 | ~22 mm | balanced |
| NN3 | ~30 mm | cortical |
| NN4 | ~36 mm | cortical, near noise floor |

## 2. Tissue optical properties

Per wavelength, baseline absorption is built from baseline haemoglobin concentrations
(C_HbO ≈ 56 µM, C_HbR ≈ 24 µM; total ≈ 80 µM, SO2 ≈ 70 %):

```
mu_a(λ) = ln(10) · (eps_HbO(λ)·C_HbO + eps_HbR(λ)·C_HbR) + mu_a_background(λ)     [1/cm]
mu_s'(λ) = mu_s'(800 nm) · (λ / 800)^(-b),   mu_s'(800) ≈ 11 /cm,  b ≈ 1.3        [1/cm]
```

This gives mu_a ≈ 0.17 /cm (750) and ≈ 0.20 /cm (850), mu_s' ≈ 12 and ≈ 10 /cm - plausible
adult-brain NIR values.

## 3. Light fall-off - Farrell (1992) diffuse reflectance

Detected intensity vs separation `rho` uses the steady-state spatially-resolved diffuse
reflectance for a semi-infinite medium with an extrapolated boundary:

```
mu_t' = mu_a + mu_s'
D     = 1 / (3·mu_t')
mu_eff= sqrt(mu_a / D) = sqrt(3·mu_a·mu_t')
z0    = 1 / mu_t'
A     = (1 + R_eff) / (1 - R_eff),   R_eff from the n = 1.4 polynomial (≈ 0.43)
zb    = 2·A·D
r1    = sqrt(z0^2 + rho^2)
r2    = sqrt((z0 + 2·zb)^2 + rho^2)

R(rho) = (1/4π) [ z0·(mu_eff + 1/r1)·exp(-mu_eff·r1)/r1^2
                + (z0 + 2·zb)·(mu_eff + 1/r2)·exp(-mu_eff·r2)/r2^2 ]
```

`R(rho)` is normalised so NN1 ≈ 1, scattered by a log-normal coupling factor, and the
brightest short channels sit near 1 while the longest channels approach the noise floor.

## 4. Modified Beer-Lambert law (forward)

Haemoglobin changes are projected to a **natural-log** change in optical density per
wavelength and per channel (distance `d` in cm):

```
ΔOD(λ) = ln(10) · DPF(λ) · d · ( eps_HbO(λ)·ΔHbO + eps_HbR(λ)·ΔHbR )
```

The `ln(10)` converts the base-10 extinction coefficients (cm⁻¹·M⁻¹) to the natural-log OD
that the pipeline recovers from `-ln(I / I_baseline)`. Detected intensity is then
`I = I0 · exp(-ΔOD)` plus shot + detector-floor noise.

### Extinction coefficients (Prahl / OMLC, cm⁻¹·M⁻¹, base-10)

| λ | eps_HbO | eps_HbR |
|---|---|---|
| 750 nm | 518 | 1405.24 |
| 850 nm | 1058 | 691.32 |

The isosbestic point is ~800 nm: HbR absorbs more below it, HbO more above it.

### DPF (adult head)

DPF(750) ≈ 6.4, DPF(850) ≈ 5.75.

## 5. Modified Beer-Lambert law (inverse)

With two wavelengths the 2×2 system is invertible. Per channel/time:

```
y(λ)        = ΔOD(λ) / ( ln(10) · DPF(λ) · d )
[ΔHbO,ΔHbR] = E⁻¹ · [y(750), y(850)],   E = [[eps_HbO(750), eps_HbR(750)],
                                             [eps_HbO(850), eps_HbR(850)]]
```

A noise-free forward → inverse round-trip recovers the input concentrations to machine
precision. Because MBLL assumes a single homogeneous compartment, recovered HbO/HbR mix
cortical and scalp contributions - the motivation for short-separation regression.

## 6. Quality control

- **Pruning** (NeuroDOT `findGoodMeas`-style): keep a measurement if its mean light is in
  bounds and its temporal standard deviation of OD is between a small floor (flatline guard)
  and **7.5 %**.
- **Cardiac SNR / SCI**: power in the ~1 Hz band vs high-frequency noise - a scalp-coupling
  indicator, shown as a quality map (not a hard gate).
- **GVTD**: RMS across channels of the frame-to-frame OD derivative, thresholded at
  `median + 5·1.4826·MAD` (robust) - a motion detector.

## 7. Preprocessing

- **Band-pass**: zero-phase Butterworth, 0.02-1 Hz (`scipy.signal.butter` + `filtfilt`),
  removing slow drift and the cardiac band while keeping the hemodynamic band.
- **Short-separation regression**: estimate the scalp signal as the mean of the good NN1
  (≤ 20 mm) channels and regress it out of every channel by least squares.

## References

- T. J. Farrell, M. S. Patterson, B. Wilson, "A diffusion theory model of spatially resolved,
  steady-state diffuse reflectance for the noninvasive determination of tissue optical
  properties in vivo," *Medical Physics* 19(4), 879-888 (1992).
- S. Prahl, "Optical absorption of hemoglobin," Oregon Medical Laser Center (OMLC tabulation).
- A. T. Eggebrecht et al., NeuroDOT toolbox (QC-plot conventions and processing pipeline).
