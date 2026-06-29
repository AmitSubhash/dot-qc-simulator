import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import math

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "font.size": 9,
    })

    # Consistent good-vs-bad colour language used across every figure.
    GOOD_C = "0.45"      # kept channels: neutral grey
    BAD_C = "tomato"     # pruned channels: red
    HBO_C = "tab:red"    # oxy-haemoglobin
    HBR_C = "tab:blue"   # deoxy-haemoglobin
    return BAD_C, GOOD_C, HBO_C, HBR_C, butter, filtfilt, math, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # DOT / fNIRS Signal Quality Simulator

        > **All controls are pinned in the left sidebar** (toggle it with the sidebar
        > icon, top-left) so the preset and sliders stay next to whatever plot you are
        > viewing - no scrolling back up.

        A visual sandbox for the quality-control plots you meet in high-density
        fNIRS / NeuroDOT lectures (built alongside Ari Segel's *NeuroDOT Processing
        + Analysis* walkthrough). It generates **physics-based** synthetic
        two-wavelength (750 nm + 850 nm) data and lets you add artifacts one slider
        at a time, then watch how each QC plot reacts.

        Unlike a pure cartoon, the signal chain here is grounded in the real
        forward model:

        - A **checkerboard source/detector montage** produces measurements at
          discrete nearest-neighbour separations (~13, 29, 39, 47 mm), exactly the
          NN1-NN4 bands you see in NeuroDOT fall-off plots.
        - Detected light follows the **Farrell (1992) semi-infinite diffuse
          reflectance** solution, so intensity falls ~5-6 orders of magnitude with
          separation down to a noise floor.
        - Brain and scalp **haemoglobin changes (HbO, HbR)** are projected to
          optical density at each wavelength through the **modified Beer-Lambert
          law** using the real Prahl/omlc extinction coefficients. The app then
          **inverts** that law to recover HbO/HbR, just like a real pipeline.

        **The goal is still pattern recognition:** what do motion, drift, poor
        coupling, systemic physiology, bad channels, and source-detector distance
        *look like* on these plots, and what does preprocessing actually fix?

        ### The one idea that ties every plot together

        A measurement's **raw light quality** and its **neural-response quality**
        are not the same thing:

        - **Short separation (NN1, ~13 mm)**: lots of light, a clean cardiac pulse,
          beautiful traces. But the banana-shaped photon path barely reaches cortex,
          so the cortical response is tiny. Short channels are excellent
          *regressors of scalp physiology*, poor *cortical* channels.
        - **Long separation (NN3, ~39 mm)**: the photon path dips into cortex, so
          these carry the **most brain signal** - but far less light survives, SNR
          is worse, and they prune first. The cortical workhorse *when they survive*.
        - **Mid (NN2, ~29 mm)** is the usual compromise.

        That trade-off is why QC is layered: a channel can pass "do we have light?"
        and still fail "can we trust the brain signal?"
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ## What should I be looking at?

            Each plot has a healthy "tell" and a warning "tell". Scan top to bottom:

            | Plot | Healthy run shows | Worry when you see |
            |---|---|---|
            | **Raw light** | stable bands, a heartbeat in the zoom, ~3/4 kept | bands crashing to the floor; lots pruned |
            | **QC metrics** | most channels left of the 7.5% line; high cardiac SNR for short | a fat tail past the std line; SNR near 0 dB |
            | **Cap view** | mostly green links, coverage everywhere | red links wiping out a whole region |
            | **OD traces / grayplot** | calm grey, gentle physiology | vertical stripes (motion) or whole bright/dark rows (bad channels) |
            | **GVTD** | a flat low line | spikes above the dashed threshold (motion frames) |
            | **HbO / HbR** | HbO **up**, HbR **down**, growing short -> mid -> long | flat response, or HbR not mirroring HbO |

            ### Try these (and watch the named plot)

            1. Preset **Motion spikes** -> *GVTD* spikes over threshold and the *grayplot* shows vertical stripes.
            2. Preset **Bad optode coupling** -> *raw-light* channels drop toward the noise floor and the *cap view* turns red.
            3. Preset **Strong superficial physiology**, then flip on **NN1 regression** -> in *Preprocessing effect*, the long-channel HbO bump shrinks to its true cortical size (the scalp part is removed).
            4. Preset **Pulse/respiration**, then flip on **Band-pass** -> the cardiac ripple riding on the *HbO/HbR* trace disappears.
            5. In the **HbO/HbR** plot, compare the *short* vs *long* panels -> short looks cleanest yet barely responds (it is scalp); long responds most but is noisiest (it is cortex).

            *Where preprocessing shows:* the **band-pass** and **NN1 regression** toggles reshape only the **OD traces, grayplot, and HbO/HbR** plots - not the raw-light, QC-metric, or cap plots (those are raw on purpose). If a toggle looks like it did nothing, scroll to the **Preprocessing effect** plot: it always shows raw -> filtered -> regressed side by side. At the *Clean good data* preset the effect is small (little to remove); switch to *Strong superficial physiology* or *Pulse/respiration* to see it clearly.

            *Tip:* turn on **Compare clean vs corrupted** to see any artifact side-by-side against an artifact-free version of the same run.
            """
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "Glossary (click to expand)": mo.md(
                """
                - **Raw light / intensity** - detected photons at the optode, before
                  processing. Should sit above the noise floor, below saturation, and
                  ideally show a visible heartbeat.
                - **Nearest neighbour (NN1-NN4)** - measurement groups by
                  source-detector separation. NN1 ~ shortest/brightest, NN4 ~ longest/
                  dimmest. They appear as vertical bands in the fall-off plot.
                - **Optical density (OD, Delta OD)** - `-log(I / I_baseline)`, the
                  relative attenuation change DOT reconstructs.
                - **Modified Beer-Lambert law (MBLL)** - `Delta OD(lambda) = ln(10) *
                  DPF(lambda) * d * (eps_HbO * Delta[HbO] + eps_HbR * Delta[HbR])`.
                  Two wavelengths give two equations, so we can solve for HbO and HbR.
                - **Extinction coefficient (eps)** - how strongly HbO/HbR absorb at a
                  wavelength. At 750 nm HbR dominates; at 850 nm HbO dominates
                  (crossover ~800 nm). Real Prahl/omlc values are used here.
                - **DPF** - differential pathlength factor: scattering makes photons
                  travel farther than the straight source-detector distance.
                - **HbO / HbR** - oxy- and deoxy-haemoglobin concentration changes.
                  A cortical response is HbO **up**, HbR **down**.
                - **Diffuse reflectance** - the analytic light fall-off with distance
                  (Farrell 1992), set by tissue absorption and scattering.
                - **Temporal std (pruning metric)** - NeuroDOT keeps a measurement if
                  the temporal standard deviation of its OD is below ~7.5%.
                - **Cardiac SNR / SCI** - power near 1 Hz vs high-frequency noise; a
                  visible pulse means good scalp coupling. Shown as a quality map.
                - **GVTD** - global variance of the temporal derivative; spikes when
                  many channels jump together (motion detector).
                - **Grayplot** - channels x time image; reveals global events and bad rows.
                - **Block average** - averaging stimulus-locked windows to pull the
                  response out of noise.
                - **Band-pass filter** - here 0.02-1 Hz: removes slow drift and the
                  fast cardiac pulse, keeping the hemodynamic band.
                - **Superficial regression** - estimate scalp physiology from NN1
                  (short) channels and subtract it from longer channels. This app
                  *does* perform it (toggle below).
                """
            )
        }
    )
    return


@app.cell
def _(mo):
    preset = mo.ui.dropdown(
        options=[
            "Clean good data",
            "Motion spikes",
            "Baseline shift",
            "Bad optode coupling",
            "Strong superficial physiology",
            "Pulse/respiration contamination",
            "Drift-dominated run",
            "Too many bad channels",
            "Wavelength imbalance",
        ],
        value="Clean good data",
        label="Preset",
    )
    seed = mo.ui.slider(0, 999, value=42, step=1, label="Random seed")
    compare = mo.ui.switch(value=False, label="Compare clean vs corrupted")
    bandpass_on = mo.ui.switch(value=False, label="Band-pass 0.02-1 Hz")
    regress_on = mo.ui.switch(value=False, label="Short-separation (NN1) regression")

    white_noise = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra white detector noise")
    drift = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra low-frequency drift")
    pulse = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra pulse ~1 Hz")
    respiration = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra respiration ~0.25 Hz")
    mayer = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra Mayer/systemic ~0.1 Hz")
    superficial = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra shared superficial physiology")
    motion_spikes = mo.ui.slider(0.0, 5.0, value=0.0, step=0.5, label="Extra motion spikes")
    baseline_shift = mo.ui.slider(0.0, 5.0, value=0.0, step=0.5, label="Extra baseline shifts")
    bad_channels = mo.ui.slider(0.0, 0.45, value=0.0, step=0.05, label="Extra bad-channel fraction")
    poor_coupling = mo.ui.slider(0.0, 0.45, value=0.0, step=0.05, label="Extra poor-coupling fraction")
    wavelength_imbalance = mo.ui.slider(0.0, 3.0, value=0.0, step=0.1, label="Extra wavelength imbalance")

    # Pin all controls into a left sidebar so they stay beside whatever graph you
    # are looking at (no scrolling back up). Sliders live in collapsible sections
    # to keep the sidebar compact; preset + preprocessing switches stay on top.
    mo.sidebar(
        [
            mo.md("## Controls"),
            preset,
            seed,
            mo.md("**Preprocessing**"),
            bandpass_on,
            regress_on,
            compare,
            mo.md("---\n**Artifact sliders** *(add on top of the preset; each uses an independent random stream)*"),
            mo.accordion(
                {
                    "Noise & physiology": mo.vstack(
                        [white_noise, drift, pulse, respiration, mayer, superficial]
                    ),
                    "Channel faults & imbalance": mo.vstack(
                        [motion_spikes, baseline_shift, bad_channels, poor_coupling, wavelength_imbalance]
                    ),
                },
                multiple=True,
            ),
        ],
        width=340,
    )
    return (
        bad_channels,
        bandpass_on,
        baseline_shift,
        compare,
        drift,
        mayer,
        motion_spikes,
        poor_coupling,
        preset,
        pulse,
        regress_on,
        respiration,
        seed,
        superficial,
        wavelength_imbalance,
        white_noise,
    )


@app.cell
def _(
    bad_channels,
    baseline_shift,
    drift,
    mayer,
    motion_spikes,
    poor_coupling,
    preset,
    pulse,
    respiration,
    seed,
    superficial,
    wavelength_imbalance,
    white_noise,
):
    PRESETS = {
        "Clean good data": dict(),
        "Motion spikes": dict(motion_spikes=3.0),
        "Baseline shift": dict(baseline_shift=3.0),
        "Bad optode coupling": dict(poor_coupling=0.25, white_noise=1.4),
        "Strong superficial physiology": dict(superficial=2.2, mayer=1.3),
        "Pulse/respiration contamination": dict(pulse=2.2, respiration=2.2),
        "Drift-dominated run": dict(drift=2.4),
        "Too many bad channels": dict(bad_channels=0.30, poor_coupling=0.18),
        "Wavelength imbalance": dict(wavelength_imbalance=2.2, white_noise=0.8),
    }
    # Baseline of a healthy run: a real heartbeat present (coupling sign), rest low.
    base = dict(
        white_noise=1.0,
        drift=0.5,
        pulse=0.7,
        respiration=0.5,
        mayer=0.5,
        superficial=0.7,
        motion_spikes=0.0,
        baseline_shift=0.0,
        bad_channels=0.02,
        poor_coupling=0.03,
        wavelength_imbalance=0.0,
    )
    base.update(PRESETS[preset.value])
    params = {
        "seed": int(seed.value),
        "white_noise": base["white_noise"] + white_noise.value,
        "drift": base["drift"] + drift.value,
        "pulse": base["pulse"] + pulse.value,
        "respiration": base["respiration"] + respiration.value,
        "mayer": base["mayer"] + mayer.value,
        "superficial": base["superficial"] + superficial.value,
        "motion_spikes": base["motion_spikes"] + motion_spikes.value,
        "baseline_shift": base["baseline_shift"] + baseline_shift.value,
        "bad_channels": min(0.75, base["bad_channels"] + bad_channels.value),
        "poor_coupling": min(0.75, base["poor_coupling"] + poor_coupling.value),
        "wavelength_imbalance": base["wavelength_imbalance"] + wavelength_imbalance.value,
    }
    # Pristine reference for comparison mode: keep heartbeat + neural, drop artifacts.
    clean_params = {
        "seed": int(seed.value),
        "white_noise": 0.7, "drift": 0.1, "pulse": 0.7, "respiration": 0.3,
        "mayer": 0.2, "superficial": 0.35, "motion_spikes": 0.0, "baseline_shift": 0.0,
        "bad_channels": 0.0, "poor_coupling": 0.0, "wavelength_imbalance": 0.0,
    }
    return clean_params, params


@app.cell
def _(mo, preset):
    notes = {
        "Clean good data": """
        **What to look for:** clean NN bands in the fall-off, a visible heartbeat on
        short/mid traces, ~3/4 of channels kept, low flat GVTD, and an HbO/HbR
        response (HbO up, HbR down) that *grows* from short -> mid -> long. Short
        channels look cleanest yet carry the weakest cortical response.
        """,
        "Motion spikes": """
        **What to look for:** vertical stripes across traces and the grayplot, and
        sharp **GVTD** spikes above threshold (gold). GVTD *detects* bad frames; it
        does not fix them. Pruning is barely affected - motion ruins *frames*, not
        whole channels. Watch the spikes inject into recovered HbO/HbR.
        """,
        "Baseline shift": """
        **What to look for:** step changes after cap/optode movement, subtle on
        log-light plots but obvious as a single **GVTD** spike and a brightness step
        in the **grayplot**. Baseline subtraction becomes unreliable after the step.
        """,
        "Bad optode coupling": """
        **What to look for:** low-light, high-variance channels with **no heartbeat**.
        Temporal std climbs above 7.5% -> pruned. Long channels collapse first. The
        HbO/HbR recovery loses its long-distance contributors.
        """,
        "Strong superficial physiology": """
        **What to look for:** large shared scalp signal, strongest in **short**
        channels. It does **not** prune the channels - it contaminates HbO/HbR. Turn
        on **short-separation regression** and watch the systemic component vanish
        from the longer channels. This is why NN1 regressors exist.
        """,
        "Pulse/respiration contamination": """
        **What to look for:** rhythmic wiggles and sharp **spectrum** peaks at ~1 Hz
        and ~0.25 Hz, and a strong cardiac ripple riding on the HbO/HbR block average.
        Turn on the **band-pass** filter to remove the pulse and reveal the response.
        """,
        "Drift-dominated run": """
        **What to look for:** slow wander over the run and a rising low-frequency tail
        in the **spectrum**. Drift inflates temporal std, so heavy drift can start
        pruning channels. The **band-pass** high-pass removes it.
        """,
        "Too many bad channels": """
        **What to look for:** flatline / saturated / noise-floor faults remove a big
        fraction of measurements (red in the cap view). Even if survivors look fine,
        **spatial coverage** may be too sparse for a trustworthy image.
        """,
        "Wavelength imbalance": """
        **What to look for:** 850 nm has worse light/SNR than 750 nm. Because MBLL
        needs **both** wavelengths to solve for HbO/HbR, one weak wavelength corrupts
        the recovered concentrations even where the other looks perfect.
        """,
    }
    mo.callout(mo.md(notes[preset.value]), kind="warn")
    return


@app.cell
def _(butter, filtfilt, math, np):
    # --- timing-independent helpers -------------------------------------------
    def gamma_pdf(t, k, theta):
        out = np.zeros_like(t, dtype=float)
        pos = t > 0
        tp = t[pos]
        out[pos] = tp ** (k - 1) * np.exp(-tp / theta) / (theta**k * math.gamma(k))
        return out


    def double_gamma_hrf(t):
        """SPM-style canonical HRF: positive peak ~6 s, undershoot ~16 s."""
        h = gamma_pdf(t, 6.0, 1.0) - (1.0 / 6.0) * gamma_pdf(t, 16.0, 1.0)
        return h / (np.max(np.abs(h)) + 1e-12)


    def block_design(t, start=20.0, on=20.0, off=20.0):
        cycle = on + off
        active = (t >= start) & (((t - start) % cycle) < on)
        onsets = np.arange(start, t[-1] - on, cycle)
        return active.astype(float), onsets


    def robust_z(x, axis=None):
        med = np.nanmedian(x, axis=axis, keepdims=True)
        mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
        return (x - med) / (1.4826 * mad + 1e-9)


    def power_spectrum(signal, fs):
        y = signal - np.mean(signal)
        freqs = np.fft.rfftfreq(y.size, d=1 / fs)
        power = np.abs(np.fft.rfft(y)) ** 2
        return freqs, power / (power.max() + 1e-12)

    # --- diffuse-optics physics -----------------------------------------------
    # Hemoglobin molar extinction coefficients [cm^-1 / M], base-10 (Prahl/omlc;
    # matches the canonical Homer2 set). Isosbestic ~800 nm.
    EXT = {750: {"HbO": 518.0, "HbR": 1405.24}, 850: {"HbO": 1058.0, "HbR": 691.32}}
    DPF = {750: 6.4, 850: 5.75}        # differential pathlength factor (adult head)
    C_HBO0, C_HBR0 = 56e-6, 24e-6      # baseline conc [M] (total Hb 80 uM, SO2 ~70%)
    MUA_BG = {750: 0.025, 850: 0.030}  # background absorption [1/cm]
    MUSP_800, MUSP_B = 11.0, 1.3       # reduced scattering musp(800nm), wavelength slope
    N_TISSUE = 1.4


    def optical_props(wl):
        """Wavelength-specific mu_a and mu_s' [1/cm] from baseline haemoglobin."""
        mua = math.log(10) * (EXT[wl]["HbO"] * C_HBO0 + EXT[wl]["HbR"] * C_HBR0) + MUA_BG[wl]
        musp = MUSP_800 * (wl / 800.0) ** (-MUSP_B)
        return mua, musp


    def farrell_reflectance(rho_cm, mua, musp):
        """Steady-state diffuse reflectance, semi-infinite medium (Farrell 1992)."""
        mut = mua + musp
        D = 1.0 / (3.0 * mut)
        mueff = np.sqrt(mua / D)
        z0 = 1.0 / mut
        reff = -1.440 / N_TISSUE**2 + 0.710 / N_TISSUE + 0.668 + 0.0636 * N_TISSUE
        A = (1 + reff) / (1 - reff)
        zb = 2 * A * D
        r1 = np.sqrt(z0**2 + rho_cm**2)
        r2 = np.sqrt((z0 + 2 * zb) ** 2 + rho_cm**2)
        return (1 / (4 * np.pi)) * (
            z0 * (mueff + 1 / r1) * np.exp(-mueff * r1) / r1**2
            + (z0 + 2 * zb) * (mueff + 1 / r2) * np.exp(-mueff * r2) / r2**2
        )


    def mbll_inverse_matrix():
        """Inverse of [[eHbO750,eHbR750],[eHbO850,eHbR850]] for MBLL recovery."""
        E = np.array([[EXT[750]["HbO"], EXT[750]["HbR"]],
                      [EXT[850]["HbO"], EXT[850]["HbR"]]])
        return np.linalg.inv(E)


    def bandpass_filter(data, fs, lo=0.02, hi=1.0, order=4):
        b, a = butter(order, [lo, hi], btype="band", fs=fs)
        return filtfilt(b, a, data, axis=-1)
    return (
        DPF,
        EXT,
        bandpass_filter,
        block_design,
        double_gamma_hrf,
        farrell_reflectance,
        mbll_inverse_matrix,
        optical_props,
        power_spectrum,
        robust_z,
    )


@app.cell
def _(
    DPF,
    EXT,
    block_design,
    clean_params,
    compare,
    double_gamma_hrf,
    farrell_reflectance,
    math,
    np,
    optical_props,
    params,
):
    # --- fixed simulation constants -------------------------------------------
    FS = 10.0
    DURATION = 300.0
    WAVELENGTHS = (750, 850)

    # Real cap geometry: the Piglet 26-source / 31-detector pad
    # (NeuroDOT Pad_Piglet26s31d_x10mm.mat, spos2/dpos2 in mm). Nearest-neighbour
    # separations are 10, 22, 30, 36 mm (NN1-NN4).
    _SRC_XY = [
        (-21.2, -35.4), (-7.1, -35.4), (7.1, -35.4), (21.2, -35.4),
        (-21.2, -21.2), (-7.1, -21.2), (7.1, -21.2), (21.2, -21.2),
        (-21.2, -7.1), (-7.1, -7.1), (7.1, -7.1), (21.2, -7.1),
        (-21.2, 7.1), (-7.1, 7.1), (7.1, 7.1), (21.2, 7.1),
        (-21.2, 21.2), (-7.1, 21.2), (7.1, 21.2), (21.2, 21.2),
        (-21.2, 35.4), (-7.1, 35.4), (7.1, 35.4), (21.2, 35.4),
        (-7.1, 49.5), (7.1, 49.5),
    ]
    _DET_XY = [
        (-14.1, -42.4), (0.0, -42.4), (14.1, -42.4),
        (-28.3, -28.3), (-14.1, -28.3), (0.0, -28.3), (14.1, -28.3), (28.3, -28.3),
        (-28.3, -14.1), (-14.1, -14.1), (0.0, -14.1), (14.1, -14.1), (28.3, -14.1),
        (-28.3, 0.0), (-14.1, 0.0), (0.0, 0.0), (14.1, 0.0), (28.3, 0.0),
        (-28.3, 14.1), (-14.1, 14.1), (0.0, 14.1), (14.1, 14.1), (28.3, 14.1),
        (-28.3, 28.3), (-14.1, 28.3), (0.0, 28.3), (14.1, 28.3), (28.3, 28.3),
        (-14.1, 42.4), (0.0, 42.4), (14.1, 42.4),
    ]
    MAX_RSD = 40.0         # keep S-D pairs out to NN4 (the lecture's "Rsd 1-40 mm")
    NN1_MM = 10.0          # first-nearest-neighbour distance (light normalisation)

    DET_FLOOR = 1.5e-5     # detector noise floor (sets the light floor; NN4 straddles it)
    SHOT_COEFF = 5.0e-4    # shot-like noise, grows with sqrt(intensity)

    # Hemodynamic amplitudes [M].
    HBO_NEURAL, HBR_NEURAL = 0.55e-6, -0.21e-6   # cortical response
    HBO_SYS, HBR_SYS = 0.30e-6, 0.10e-6          # ongoing systemic/scalp oscillation
    HBO_TASK, HBR_TASK = 0.24e-6, -0.07e-6       # task-locked SCALP response (the
    #                                              artifact short-channel regression removes

    # Pruning gates. Primary = temporal std of OD < 7.5% (NeuroDOT findGoodMeas).
    STD_THRESH, STD_MIN = 0.075, 0.001
    SCI_THRESH = 1.5       # cardiac-SNR reference line for the coupling map only
    LIGHT_FLOOR, LIGHT_CEIL = 3e-5, 4.0
    GVTD_K = 5.0


    def _build_montage():
        src = np.array(_SRC_XY, dtype=float)
        det = np.array(_DET_XY, dtype=float)
        pairs = []
        for s in src:
            for d in det:
                rsd = float(np.hypot(*(s - d)))
                if rsd <= MAX_RSD:
                    pairs.append((rsd, 0.5 * (s + d), s, d))
        return src, det, pairs


    def _group_label(rsd_mm):
        # NN1~10 mm, NN2~22 mm, NN3~30 mm, NN4~36 mm
        if rsd_mm < 16:
            return "short <20 mm"
        if rsd_mm < 26:
            return "mid 20-30 mm"
        return "long 30-40 mm"


    def simulate_dot(p):
        # independent substreams -> orthogonal, reproducible artifacts
        s = p["seed"]
        rng = np.random.default_rng(s)
        rng_phys = np.random.default_rng(s + 1)
        rng_noise = np.random.default_rng(s + 2)
        rng_motion = np.random.default_rng(s + 3)
        rng_base = np.random.default_rng(s + 4)

        t = np.arange(0, DURATION, 1 / FS)
        n_t = t.size

        src, det, pairs = _build_montage()
        rsd_pair = np.array([pr[0] for pr in pairs])
        mid_pair = np.array([pr[1] for pr in pairs])
        src_pair = np.array([pr[2] for pr in pairs])
        det_pair = np.array([pr[3] for pr in pairs])
        n_pair = len(pairs)

        # depth sensitivity: cortex seen more by long channels, scalp by all
        cortical_sens = 1.0 / (1.0 + np.exp(-(rsd_pair - 24.0) / 5.0))
        scalp_sens = np.clip(1.2 - 0.004 * rsd_pair, 0.6, 1.2)

        # neural drive + systemic (scalp) physiology, as concentration time courses
        stim, onsets = block_design(t)
        hrf = double_gamma_hrf(np.arange(0, 35, 1 / FS))
        neural = np.convolve(stim, hrf, mode="full")[:n_t] / FS
        neural /= np.max(np.abs(neural)) + 1e-12

        # task-locked SCALP hemodynamics: a superficial response to the stimulus with
        # a faster, earlier shape than cortex (time-compressed HRF). It survives block
        # averaging and masquerades as activation, so it is exactly what short-
        # separation (NN1) regression is designed to remove.
        scalp_hrf = double_gamma_hrf(np.arange(0, 35, 1 / FS) * 1.8)
        scalp_task = np.convolve(stim, scalp_hrf, mode="full")[:n_t] / FS
        scalp_task /= np.max(np.abs(scalp_task)) + 1e-12

        pulse = np.sin(2 * np.pi * (1.05 + rng_phys.normal(0, 0.02)) * t + rng_phys.uniform(0, 2 * np.pi))
        resp = np.sin(2 * np.pi * 0.27 * t + rng_phys.uniform(0, 2 * np.pi))
        mayer = np.sin(2 * np.pi * 0.095 * t + rng_phys.uniform(0, 2 * np.pi))
        drift = np.sin(2 * np.pi * 0.006 * t + rng_phys.uniform(0, 2 * np.pi)) + 0.6 * (t / t[-1] - 0.5)
        systemic = (0.7 * p["pulse"] * pulse + 0.5 * p["respiration"] * resp
                    + 0.5 * p["mayer"] * mayer + 0.5 * p["drift"] * drift)

        # cortical response lives at depth (cortical_sens); systemic + task-locked
        # scalp signals live superficially (scalp_sens) and scale with the
        # "superficial" control. The scalp task response is shared by short and long
        # channels alike - which is why a short channel can show a "response".
        scalp_HbO = scalp_sens[:, None] * p["superficial"] * (
            HBO_SYS * systemic[None, :] + HBO_TASK * scalp_task[None, :])
        scalp_HbR = scalp_sens[:, None] * p["superficial"] * (
            HBR_SYS * systemic[None, :] + HBR_TASK * scalp_task[None, :])
        dC_HbO = cortical_sens[:, None] * HBO_NEURAL * neural[None, :] + scalp_HbO
        dC_HbR = cortical_sens[:, None] * HBR_NEURAL * neural[None, :] + scalp_HbR

        # shared optode-motion perturbations (affect both wavelengths of a pair)
        motion_od = np.zeros((n_pair, n_t))
        for _ in range(int(round(p["motion_spikes"] * 3))):
            center = rng_motion.integers(int(15 * FS), n_t - int(15 * FS))
            width = rng_motion.integers(int(0.4 * FS), int(1.5 * FS))
            affected = rng_motion.random(n_pair) < rng_motion.uniform(0.35, 0.90)
            amp = rng_motion.normal(0.05, 0.02) * rng_motion.choice([-1, 1])
            transient = amp * np.exp(-np.arange(width) / (0.25 * FS))
            end = min(n_t, center + width)
            motion_od[affected, center:end] += transient[: end - center]
        for _ in range(int(round(p["baseline_shift"] * 2))):
            center = rng_base.integers(int(30 * FS), n_t - int(30 * FS))
            affected = rng_base.random(n_pair) < rng_base.uniform(0.15, 0.50)
            amp = rng_base.normal(0.03, 0.012) * rng_base.choice([-1, 1])
            motion_od[affected, center:] += amp

        # MBLL forward -> OD per wavelength, then to detected intensity
        wl_list, raw_list, od_list = [], [], []
        ml_list, std_list, sci_list, good_list = [], [], [], []
        for wl in WAVELENGTHS:
            od_true = (math.log(10) * DPF[wl] * (rsd_pair[:, None] / 10.0)
                       * (EXT[wl]["HbO"] * dC_HbO + EXT[wl]["HbR"] * dC_HbR)) + motion_od

            mua, musp = optical_props(wl)
            refl = farrell_reflectance(rsd_pair / 10.0, mua, musp)
            refl_nn1 = farrell_reflectance(np.array([NN1_MM / 10.0]), mua, musp)[0]
            i0 = refl / refl_nn1 * rng.lognormal(0.0, 0.25, size=n_pair)
            i0 *= 1.0 if wl == 750 else 1.0 / (1 + 0.3 * p["wavelength_imbalance"])
            raw = i0[:, None] * np.exp(-od_true)

            noise = (DET_FLOOR + SHOT_COEFF * np.sqrt(np.abs(raw))) * p["white_noise"]
            noise *= 1.0 if wl == 750 else (1 + 0.6 * p["wavelength_imbalance"])
            raw = raw + rng_noise.standard_normal(raw.shape) * noise

            # poor coupling: big light drop + extra noise
            poor = rng.random(n_pair) < p["poor_coupling"]
            if poor.any():
                raw[poor] *= rng.uniform(0.02, 0.10, size=(poor.sum(), 1))
                raw[poor] += rng_noise.normal(0, DET_FLOOR, size=(poor.sum(), n_t))

            # hardware faults
            bad = rng.random(n_pair) < p["bad_channels"]
            bad_type = rng.choice(["flatline", "saturated", "noise_floor"], size=n_pair)
            for i in np.where(bad)[0]:
                if bad_type[i] == "flatline":
                    raw[i] = np.median(raw[i])
                elif bad_type[i] == "saturated":
                    raw[i] = LIGHT_CEIL * 1.10
                else:
                    raw[i] = np.abs(rng.normal(DET_FLOOR, 0.3 * DET_FLOOR, size=n_t))

            # detectors read non-negative counts down to a noise floor; clip there
            # (not to ~0) so dead channels form a clean floor band, not deep spikes
            raw = np.clip(raw, 0.3 * DET_FLOOR, None)
            base = np.median(raw[:, : int(10 * FS)], axis=1, keepdims=True)
            od = -np.log(raw / base)

            ml = np.median(raw, axis=1)
            std_od = np.std(od, axis=1)
            centered = raw / (ml[:, None] + 1e-12) - 1
            freqs = np.fft.rfftfreq(n_t, d=1 / FS)
            spec = np.abs(np.fft.rfft(centered, axis=1)) ** 2
            card = (freqs >= 0.7) & (freqs <= 1.5)
            high = (freqs >= 2.2) & (freqs <= 4.5)
            sci = 10 * np.log10((spec[:, card].mean(1) + 1e-18) / (spec[:, high].mean(1) + 1e-18))
            good = ((ml > LIGHT_FLOOR) & (ml < LIGHT_CEIL) & (std_od > STD_MIN) & (std_od < STD_THRESH))

            wl_list.append(np.full(n_pair, wl))
            raw_list.append(raw)
            od_list.append(od)
            ml_list.append(ml)
            std_list.append(std_od)
            sci_list.append(sci)
            good_list.append(good)

        # flat channel arrays: first n_pair rows are 750 nm, next n_pair are 850 nm
        wl = np.concatenate(wl_list)
        rsd = np.tile(rsd_pair, 2)
        group = np.array([_group_label(r) for r in rsd])
        pair_id = np.tile(np.arange(n_pair), 2)
        raw = np.vstack(raw_list)
        od = np.vstack(od_list)
        mean_light = np.concatenate(ml_list)
        std_od = np.concatenate(std_list)
        sci_db = np.concatenate(sci_list)
        good = np.concatenate(good_list)
        midpoint = np.vstack([mid_pair, mid_pair])
        source_xy = np.vstack([src_pair, src_pair])
        detector_xy = np.vstack([det_pair, det_pair])

        # GVTD on surviving channels
        d = np.diff(od[good], axis=1)
        gvtd = np.r_[0, np.sqrt(np.mean(d**2, axis=0))] if good.any() else np.zeros(n_t)
        med_g = np.median(gvtd)
        mad_g = np.median(np.abs(gvtd - med_g))
        gvtd_threshold = med_g + GVTD_K * 1.4826 * mad_g

        return {
            "fs": FS, "t": t, "raw": raw, "od": od,
            "wl": wl, "group": group, "rsd": rsd, "pair_id": pair_id, "n_pair": n_pair,
            "rsd_pair": rsd_pair, "good": good,
            "stim": stim, "onsets": onsets, "neural": neural,
            "mean_light": mean_light, "std_od": std_od, "sci_db": sci_db,
            "gvtd": gvtd, "gvtd_threshold": gvtd_threshold, "gvtd_k": GVTD_K,
            "midpoint": midpoint, "source_xy": source_xy, "detector_xy": detector_xy,
            "src_all": src, "det_all": det,
            "light_floor": LIGHT_FLOOR, "light_ceil": LIGHT_CEIL,
            "std_thresh": STD_THRESH, "sci_thresh": SCI_THRESH,
        }

    sim = simulate_dot(params)
    sim_clean = simulate_dot(clean_params) if compare.value else None
    return sim, sim_clean


@app.cell
def _(DPF, bandpass_filter, bandpass_on, mbll_inverse_matrix, np, regress_on, sim, sim_clean):
    # Preprocessing + MBLL inversion. Returns processed OD plus recovered HbO/HbR.
    # DPF is shared with the forward model (helpers cell) so forward and inverse
    # can never desync.
    EINV = mbll_inverse_matrix()
    from math import log as _ln

    def process_run(s, do_filter, do_regress):
        n_pair = s["n_pair"]
        od = s["od"].copy()
        if do_filter:
            od = bandpass_filter(od, s["fs"])
        rsd_cm = s["rsd_pair"] / 10.0
        # split flat OD back into the two co-located wavelengths
        od750, od850 = od[:n_pair], od[n_pair:]
        y750 = od750 / (_ln(10) * DPF[750] * rsd_cm[:, None])
        y850 = od850 / (_ln(10) * DPF[850] * rsd_cm[:, None])
        dHbO = EINV[0, 0] * y750 + EINV[0, 1] * y850
        dHbR = EINV[1, 0] * y750 + EINV[1, 1] * y850
        good_pair = s["good"][:n_pair] & s["good"][n_pair:]
        if do_regress:
            nn1 = (s["rsd_pair"] <= 20) & good_pair
            if nn1.any():
                for series in (dHbO, dHbR):
                    reg = series[nn1].mean(0)
                    beta = (series @ reg) / (np.dot(reg, reg) + 1e-30)
                    series -= beta[:, None] * reg[None, :]
        return {"od": od, "dHbO": dHbO, "dHbR": dHbR, "good_pair": good_pair}


    proc = process_run(sim, bandpass_on.value, regress_on.value)
    proc_clean = process_run(sim_clean, bandpass_on.value, regress_on.value) if sim_clean is not None else None
    return proc, proc_clean, process_run


@app.cell
def _(mo, np, params, sim):
    _n_total = sim["raw"].shape[0]
    _n_good = int(np.sum(sim["good"]))
    _by_group = "  ".join(
        f"{_g.split()[0]} {int(np.sum((sim['group'] == _g) & sim['good']))}/{int(np.sum(sim['group'] == _g))}"
        for _g in ["short <20 mm", "mid 20-30 mm", "long 30-40 mm"]
    )
    mo.md(
        f"""
        ### Run summary

        - Measurements kept by pruning: **{_n_good} / {_n_total}**
          ({100 * _n_good / _n_total:.0f}%) &nbsp;|&nbsp; by distance: {_by_group}
        - Montage: **{sim['src_all'].shape[0]} sources, {sim['det_all'].shape[0]} detectors,
          {sim['n_pair']} S-D pairs** x 2 wavelengths
        - Sampling **{sim["fs"]:.0f} Hz**, run length **{sim["t"][-1]:.0f} s**
        - High-GVTD (motion) frames: **{int(np.sum(sim["gvtd"] > sim["gvtd_threshold"]))}**
        - Pruning gate: temporal std of OD `< {100 * sim['std_thresh']:.1f}%`
          and light in `[{sim['light_floor']:.0e}, {sim['light_ceil']:.0f}]`
        - Active artifacts: `white_noise={params["white_noise"]:.2f}`,
          `drift={params["drift"]:.2f}`, `superficial={params["superficial"]:.2f}`,
          `motion={params["motion_spikes"]:.2f}`,
          `bad_ch={params["bad_channels"]:.2f}`,
          `poor_coupling={params["poor_coupling"]:.2f}`
        """
    )
    return


@app.cell
def _(mo, np, plt, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - raw light QC (NeuroDOT 'all vs good' view).** Each line is "
            "one measurement (coloured for legibility), log intensity over the full run. "
            "**Top row = all measurements; bottom row = only the good ones** kept by "
            "pruning - the gap between the panels is exactly what pruning removes (the "
            "fuzzy band crashing into the noise floor). Horizontal bands are the nearest-"
            "neighbour separations: brightest NN1 (short), dimmest NN3/NN4 (long). On the "
            "good panels the vertical lines are stimulus sync points - **green = task "
            "onset, red = rest onset**. (The cardiac pulse is in here too but invisible "
            "at this 300 s scale - see the zoomed view just below.) **The band-pass / "
            "regression toggles do not change this panel** - it is raw intensity; they "
            "reshape the OD traces, grayplot, and HbO/HbR plots further down."
        ),
        kind="info",
    )

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for _col, _wl in enumerate([750, 850]):
        _all = np.where(sim["wl"] == _wl)[0]
        _good = np.where((sim["wl"] == _wl) & sim["good"])[0]
        for _ax, _rows, _title in [(_axes[0, _col], _all, f"All {_wl} nm, Rsd 1-40 mm"),
                                   (_axes[1, _col], _good, f"Good {_wl} nm, Rsd 1-40 mm")]:
            _colors = plt.cm.turbo(np.linspace(0, 1, max(_rows.size, 1)))
            for _k, _i in enumerate(_rows):
                _ax.plot(sim["t"], sim["raw"][_i], lw=0.4, alpha=0.75, color=_colors[_k])
            _ax.set_yscale("log")
            _ax.set_title(_title)
            _ax.set_ylabel("Intensity")
        # stimulus sync lines on the good panel (green = task on, red = rest)
        for _o in sim["onsets"]:
            _axes[1, _col].axvline(_o, color="lime", lw=0.8, alpha=0.8)
            _axes[1, _col].axvline(_o + 20, color="red", lw=0.8, alpha=0.6)
        _axes[1, _col].set_xlabel("Time (s)")
    _fig.suptitle("Raw Light-Level QC: all measurements (top) vs good measurements (bottom)")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(mo, np, plt, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - zoomed cardiac pulse.** A 12 s window of a few good short "
            "and mid channels, each normalised to its own mean. A clear ~1 Hz "
            "**pulsatile waveform** is the hallmark of good optode-scalp coupling - it "
            "is what the cardiac-SNR map keys on. Long channels (not shown) lose this "
            "pulse into the noise, which is why they prune first. This is the 'zoomed "
            "in' view the lecture uses to confirm a heartbeat in the good data."
        ),
        kind="info",
    )
    _zoom = (sim["t"] >= 60) & (sim["t"] <= 72)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.2))
    for _ax, _grp, _title in [(_axes[0], "short <20 mm", "short / NN1"),
                              (_axes[1], "mid 20-30 mm", "mid / NN2")]:
        _sel = np.where((sim["wl"] == 850) & (sim["group"] == _grp) & sim["good"])[0][:6]
        for _i in _sel:
            _tr = sim["raw"][_i, _zoom]
            _ax.plot(sim["t"][_zoom], _tr / _tr.mean(), lw=1.0, alpha=0.85)
        _ax.set_title(f"850 nm {_title} (normalised)")
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("I / mean")
    _fig.suptitle("Zoomed Traces: the cardiac pulse marks good coupling")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(BAD_C, mo, np, plt, power_spectrum, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - QC metrics.** *Fall-off* (top-left): log light vs "
            "separation - note the discrete NN bands dropping ~5 orders to the noise "
            "floor (dashed). *Cardiac SNR* (top-right): the scalp-coupling map; long "
            "channels lose the pulse first (reference line). *Spectrum* (bottom-left): "
            "short channels carry more systemic physiology; peaks mark Mayer/resp/"
            "pulse. *Temporal std* (bottom-right): the actual pruning gate - channels "
            "right of the 7.5% line are dropped."
        ),
        kind="info",
    )

    _fig, _ax = plt.subplots(2, 2, figsize=(12, 8))
    _pruned = ~sim["good"]

    for _wl, _c in [(750, "tab:blue"), (850, "tab:orange")]:
        _m = (sim["wl"] == _wl) & sim["good"]
        _ax[0, 0].scatter(sim["rsd"][_m], sim["mean_light"][_m], s=12, alpha=0.6, color=_c, label=f"{_wl} nm")
    _ax[0, 0].scatter(sim["rsd"][_pruned], sim["mean_light"][_pruned], s=22, marker="x", color=BAD_C, label="pruned")
    _ax[0, 0].axhline(sim["light_floor"], color="0.3", ls="--", lw=0.9, label="noise floor")
    _ax[0, 0].set_yscale("log")
    _ax[0, 0].set_title("Light fall-off (nearest-neighbour bands)")
    _ax[0, 0].set_xlabel("Source-detector separation (mm)")
    _ax[0, 0].set_ylabel("Mean light")
    _ax[0, 0].legend(fontsize=7)

    _ax[0, 1].scatter(sim["rsd"][sim["good"]], sim["sci_db"][sim["good"]], s=12, alpha=0.6, color="tab:green", label="kept")
    _ax[0, 1].scatter(sim["rsd"][_pruned], sim["sci_db"][_pruned], s=22, marker="x", color=BAD_C, label="pruned")
    _ax[0, 1].axhline(sim["sci_thresh"], color="firebrick", lw=1.4, label=f"good-coupling ref {sim['sci_thresh']:.1f} dB")
    _ax[0, 1].set_title("Cardiac-band SNR (scalp coupling)")
    _ax[0, 1].set_xlabel("Source-detector separation (mm)")
    _ax[0, 1].set_ylabel("Cardiac SNR (dB)")
    _ax[0, 1].legend(fontsize=7)

    _short = (sim["wl"] == 850) & (sim["rsd"] < 20) & sim["good"]
    _long = (sim["wl"] == 850) & (sim["rsd"] >= 30) & sim["good"]
    if np.any(_short):
        _f, _p = power_spectrum(sim["raw"][_short].mean(0), sim["fs"])
        _ax[1, 0].loglog(_f[1:], _p[1:] + 1e-8, color="tab:red", label="short 850 mean")
    if np.any(_long):
        _f, _p = power_spectrum(sim["raw"][_long].mean(0), sim["fs"])
        _ax[1, 0].loglog(_f[1:], _p[1:] + 1e-8, color="tab:purple", label="long 850 mean")
    for _freq, _name in [(0.1, "Mayer"), (0.27, "resp"), (1.05, "pulse")]:
        _ax[1, 0].axvline(_freq, color="0.35", ls="--", lw=0.8)
        _ax[1, 0].text(_freq * 1.08, 0.5, _name, rotation=90, va="center", fontsize=7)
    _ax[1, 0].set_xlim(1e-3, 3)
    _ax[1, 0].set_title("Power spectrum / periodic physiology")
    _ax[1, 0].set_xlabel("Frequency (Hz)")
    _ax[1, 0].set_ylabel("Normalized power")
    _ax[1, 0].legend(fontsize=7)

    _std_pct = 100 * sim["std_od"]
    _bins = np.linspace(0, max(12, np.nanpercentile(_std_pct, 97)), 32)
    for _wl, _c in [(750, "tab:blue"), (850, "limegreen")]:
        _ax[1, 1].hist(_std_pct[sim["wl"] == _wl], bins=_bins, color=_c, alpha=0.6, label=f"{_wl} nm")
    _ax[1, 1].axvspan(100 * sim["std_thresh"], _bins[-1], color=BAD_C, alpha=0.10)
    _ax[1, 1].axvline(100 * sim["std_thresh"], color="firebrick", lw=2, label="prune > 7.5%")
    _ax[1, 1].set_title("Temporal std of OD (the pruning gate)")
    _ax[1, 1].set_xlabel("Temporal std (%)")
    _ax[1, 1].set_ylabel("Channels")
    _ax[1, 1].legend(fontsize=7)

    _fig.suptitle("NeuroDOT-Style Metrics: fall-off, coupling, spectrum, pruning gate")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(mo, np, plt, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - cap view.** The real source/detector montage. Left map = "
            "mean light, right = cardiac SNR, both falling toward longer channels. "
            "Bottom = coverage: each S-D link drawn **green if kept, red if pruned**. "
            "Watch whether red removes a whole region - that is a coverage hole no "
            "averaging can fix."
        ),
        kind="info",
    )

    _fig = plt.figure(figsize=(12, 8))
    _gs = _fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.3)
    for _i, (_title, _vals, _cmap, _lab) in enumerate([
        ("Mean light (log10)", np.log10(sim["mean_light"]), "viridis", "log10 intensity"),
        ("Cardiac SNR (dB)", sim["sci_db"], "magma", "SNR dB"),
    ]):
        _ax = _fig.add_subplot(_gs[0, _i])
        _sc = _ax.scatter(sim["midpoint"][:, 0], sim["midpoint"][:, 1], c=_vals,
                          marker="o", s=34, cmap=_cmap, edgecolor="0.25", linewidth=0.2)
        _ax.set_title(_title)
        _ax.set_aspect("equal")
        _ax.set_xticks([])
        _ax.set_yticks([])
        _fig.colorbar(_sc, ax=_ax, shrink=0.8, label=_lab)

    _ax_net = _fig.add_subplot(_gs[1, :])
    for _i in np.where(sim["rsd"] <= 42)[0]:
        _kept = sim["good"][_i]
        _xy = np.vstack([sim["source_xy"][_i], sim["detector_xy"][_i]])
        _ax_net.plot(_xy[:, 0], _xy[:, 1], color="limegreen" if _kept else "tomato",
                     alpha=0.35 if _kept else 0.7, lw=0.6 if _kept else 1.1)
    _ax_net.scatter(sim["src_all"][:, 0], sim["src_all"][:, 1], s=55, c="cornflowerblue", marker="s", label="sources", zorder=5)
    _ax_net.scatter(sim["det_all"][:, 0], sim["det_all"][:, 1], s=55, c="white", edgecolor="0.2", label="detectors", zorder=5)
    _ax_net.set_title(
        f"Measurement coverage: {int(np.sum(sim['good']))}/{sim['good'].size} kept "
        f"({100 * np.mean(sim['good']):.0f}%)  -  green kept, red pruned"
    )
    _ax_net.set_aspect("equal")
    _ax_net.set_xlabel("cap x (mm)")
    _ax_net.set_ylabel("cap y (mm)")
    _ax_net.legend(loc="upper right", frameon=True, fontsize=7)
    _fig.suptitle("Cap-Relevant QC: mean light, cardiac SNR, and spatial coverage")
    mo.vstack([_note, _fig])
    return


@app.cell
def _(bandpass_on, mo, plt, proc, regress_on, sim):
    _state = []
    if bandpass_on.value:
        _state.append("band-pass 0.02-1 Hz")
    if regress_on.value:
        _state.append("NN1 regression")
    _tag = " + ".join(_state) if _state else "no preprocessing"

    _note = mo.callout(
        mo.md(
            f"**How to read - optical density by distance.** OD = `-log(I/I_baseline)`, "
            f"currently showing **{_tag}** (toggle the switches up top). Thin lines are "
            "channels, black is the group mean, green band is the stimulus. Short "
            "channels mostly show shared scalp physiology; the stimulus-locked response "
            "emerges as you move to longer separations."
        ),
        kind="info",
    )

    _groups = ["short <20 mm", "mid 20-30 mm", "long 30-40 mm"]
    _n_pair = sim["n_pair"]
    # show 850 nm OD (processed) per group
    _od850 = proc["od"][_n_pair:]
    _good = sim["good"][_n_pair:]
    _fig, _axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for _ax, _label in zip(_axes, _groups):
        _mask = (sim["group"][_n_pair:] == _label) & _good
        _ax.plot(sim["t"], _od850[_mask].T, lw=0.5, alpha=0.3)
        if _mask.any():
            _ax.plot(sim["t"], _od850[_mask].mean(0), color="black", lw=1.6, label="group mean")
        _ax.fill_between(sim["t"], _ax.get_ylim()[0], _ax.get_ylim()[1], where=sim["stim"] > 0,
                         color="tab:green", alpha=0.08, label="stimulus")
        _ax.set_ylabel("Delta OD (850)")
        _ax.set_title(f"{_label}  ({int(_mask.sum())} good)")
        _ax.legend(loc="upper right", fontsize=7)
    _axes[-1].set_xlabel("Time (s)")
    _fig.suptitle(f"Optical-Density Traces by Distance  ({_tag})")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(mo, np, plt, proc, robust_z, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - grayplot.** Every channel is a row (sorted by wavelength "
            "then distance), time left to right, colour is robust-z OD. Calm grey = "
            "healthy. **Vertical stripes = global events** (motion, a step); **a whole "
            "bright/dark row** = a bad channel. Green lines mark stimulus onsets."
        ),
        kind="info",
    )

    _order = np.lexsort((sim["rsd"], sim["wl"]))
    _z = np.clip(robust_z(proc["od"][_order], axis=1), -5, 5)
    _fig, _ax = plt.subplots(figsize=(12, 5))
    _im = _ax.imshow(_z, aspect="auto", interpolation="nearest",
                     extent=[sim["t"][0], sim["t"][-1], _z.shape[0], 0],
                     cmap="gray", vmin=-4, vmax=4)
    _ax.set_title("Grayplot: robust-z optical density (channels x time)")
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Channels (sorted by wavelength, then distance)")
    for _onset in sim["onsets"]:
        _ax.axvline(_onset, color="lime", lw=0.8, alpha=0.5)
    _fig.colorbar(_im, ax=_ax, label="robust z")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(mo, np, plt, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - GVTD.** One number per frame: RMS change across kept "
            "channels. A **flat low line is good**. Spikes above the dashed threshold "
            "(gold) are frames where many channels jumped together = motion. Threshold "
            f"is `median + {sim['gvtd_k']:.0f} x 1.4826 x MAD` (robust). GVTD flags "
            "frames to scrub; it does not repair them."
        ),
        kind="info",
    )
    _fig, _ax = plt.subplots(figsize=(12, 3))
    _ax.plot(sim["t"], sim["gvtd"], color="firebrick", lw=1.3)
    _ax.axhline(sim["gvtd_threshold"], color="black", ls="--", lw=1.0, label="threshold")
    _high = sim["gvtd"] > sim["gvtd_threshold"]
    _ax.scatter(sim["t"][_high], sim["gvtd"][_high], color="gold", edgecolor="black", s=18, zorder=5,
                label=f"flagged frames ({int(np.sum(_high))})")
    _ax.set_title("GVTD: global variance of the temporal derivative")
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("GVTD")
    _ax.legend(loc="upper right", fontsize=7)
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(HBO_C, HBR_C, bandpass_on, mo, np, plt, proc, regress_on, sim):
    _state = []
    if bandpass_on.value:
        _state.append("band-pass")
    if regress_on.value:
        _state.append("NN1 regression")
    _tag = " + ".join(_state) if _state else "raw MBLL"

    _note = mo.callout(
        mo.md(
            "**How to read - recovered HbO/HbR (the headline fNIRS result).** The two "
            "wavelengths are inverted through the modified Beer-Lambert law into "
            "haemoglobin changes, block-averaged over stimulus repeats. A real "
            "response is **HbO up (red), HbR down (blue)**, growing short -> mid -> "
            "long with cortical sensitivity. **Careful:** short channels can show "
            "their own HbO bump - that is a *scalp* task response, not cortex. Toggle "
            "**band-pass** to kill the cardiac ripple and **NN1 regression** to "
            "subtract that scalp signal from the longer (cortical) channels."
        ),
        kind="success",
    )

    _fs = sim["fs"]
    _pre, _post = 5, 30
    _rel = np.arange(-_pre, _post, 1 / _fs)
    _groups = ["short <20 mm", "mid 20-30 mm", "long 30-40 mm"]
    _grp_pair = np.array([sim["group"][:sim["n_pair"]][i] for i in range(sim["n_pair"])])

    def _block(series, mask):
        _segs = []
        for _o in sim["onsets"]:
            _a = int((_o - _pre) * _fs)
            _b = _a + _rel.size
            if _a < 0 or _b > series.shape[1]:
                continue
            _seg = series[mask, _a:_b]
            if _seg.size:
                _segs.append(_seg - _seg[:, _rel < 0].mean(1, keepdims=True))
        if not _segs:
            return np.zeros_like(_rel)
        return np.stack(_segs, 0).mean(0).mean(0)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for _ax, _label in zip(_axes, _groups):
        _mask = (_grp_pair == _label) & proc["good_pair"]
        if _mask.any():
            _ax.plot(_rel, 1e6 * _block(proc["dHbO"], _mask), color=HBO_C, lw=2.0, label="HbO")
            _ax.plot(_rel, 1e6 * _block(proc["dHbR"], _mask), color=HBR_C, lw=2.0, label="HbR")
            _ax.axvspan(0, 20, color="tab:green", alpha=0.10)
        _ax.axhline(0, color="0.6", lw=0.6)
        _ax.axvline(0, color="black", lw=0.8)
        _ax.set_title(f"{_label}  ({int(_mask.sum())} pairs)")
        _ax.set_xlabel("Seconds from block onset")
    _axes[0].set_ylabel("Delta concentration (uM)")
    _axes[0].legend(fontsize=8, loc="upper left")
    _fig.suptitle(f"MBLL-Recovered HbO / HbR Block Average  ({_tag})")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(HBO_C, HBR_C, mo, np, plt, process_run, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - what preprocessing actually does.** Same long-distance "
            "channels, three pipelines left to right: **raw MBLL**, then **+ band-pass "
            "0.02-1 Hz**, then **+ NN1 superficial regression**. Band-pass removes the "
            "cardiac ripple and slow drift; superficial regression subtracts the scalp "
            "physiology estimated from short channels, leaving a cleaner cortical "
            "response. This is the core of the NeuroDOT preprocessing pipeline."
        ),
        kind="info",
    )

    _fs = sim["fs"]
    _pre, _post = 5, 30
    _rel = np.arange(-_pre, _post, 1 / _fs)
    _grp_pair = sim["group"][:sim["n_pair"]]
    _long = (_grp_pair == "long 30-40 mm")

    def _block(series, mask):
        _segs = []
        for _o in sim["onsets"]:
            _a = int((_o - _pre) * _fs)
            _b = _a + _rel.size
            if _a < 0 or _b > series.shape[1]:
                continue
            _seg = series[mask, _a:_b]
            if _seg.size:
                _segs.append(_seg - _seg[:, _rel < 0].mean(1, keepdims=True))
        if not _segs:
            return np.zeros_like(_rel)
        return np.stack(_segs, 0).mean(0).mean(0)

    _stages = [(False, False, "raw MBLL"), (True, False, "+ band-pass"), (True, True, "+ NN1 regression")]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for _ax, (_flt, _reg, _title) in zip(_axes, _stages):
        _pr = process_run(sim, _flt, _reg)
        _mask = _long & _pr["good_pair"]
        if _mask.any():
            _ax.plot(_rel, 1e6 * _block(_pr["dHbO"], _mask), color=HBO_C, lw=2.0, label="HbO")
            _ax.plot(_rel, 1e6 * _block(_pr["dHbR"], _mask), color=HBR_C, lw=2.0, label="HbR")
            _ax.axvspan(0, 20, color="tab:green", alpha=0.10)
        _ax.axhline(0, color="0.6", lw=0.6)
        _ax.axvline(0, color="black", lw=0.8)
        _ax.set_title(_title)
        _ax.set_xlabel("Seconds from onset")
    _axes[0].set_ylabel("Delta concentration (uM)")
    _axes[0].legend(fontsize=8, loc="upper left")
    _fig.suptitle("Preprocessing Effect on Long-Channel HbO/HbR (raw -> filtered -> regressed)")
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(mo, plt, power_spectrum, sim):
    _note = mo.callout(
        mo.md(
            "**How to read - mean-OD spectrum.** Power spectrum of the average good "
            "850 nm OD, linear axis below 1.5 Hz. Peaks at the dashed lines are "
            "systemic physiology (Mayer ~0.1, respiration ~0.27, pulse ~1 Hz). The "
            "band-pass filter (toggle up top) sits between ~0.02 and 1 Hz, so it "
            "attenuates the pulse and the slow drift while keeping the hemodynamic band."
        ),
        kind="info",
    )
    _idx = (sim["wl"] == 850) & sim["good"]
    _mean_od = sim["od"][_idx].mean(0)
    _freqs, _power = power_spectrum(_mean_od, sim["fs"])
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.plot(_freqs, _power, color="tab:purple", lw=1.4)
    for _f, _label in [(0.1, "Mayer"), (0.27, "resp"), (1.05, "pulse")]:
        _ax.axvline(_f, color="black", ls="--", lw=0.8)
        _ax.text(_f + 0.015, 0.85, _label, rotation=90, va="top", fontsize=7)
    _ax.axvspan(0.02, 1.0, color="tab:green", alpha=0.07, label="band-pass keep")
    _ax.set_xlim(0, 1.5)
    _ax.set_title("Mean-OD Spectrum: where systemic oscillations live")
    _ax.set_xlabel("Frequency (Hz)")
    _ax.set_ylabel("Normalized power")
    _ax.legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([_note, _fig])
    return


@app.cell
def _(HBO_C, mo, np, plt, proc, proc_clean, robust_z, sim, sim_clean):
    if sim_clean is None:
        _out = mo.callout(
            mo.md(
                "**Comparison mode is off.** Flip the **Compare clean vs corrupted** "
                "switch up top to render this run beside a pristine version with the "
                "*same montage and stimulus* but no artifacts - the clearest way to see "
                "what an artifact does to the grayplot, GVTD, and recovered HbO."
            ),
            kind="neutral",
        )
    else:
        _note = mo.callout(
            mo.md(
                "**How to read - clean vs corrupted.** Same seed, montage and brain "
                "response; only artifacts differ. Top: grayplots. Bottom-left: GVTD "
                "(blue clean, red current). Bottom-right: long-channel HbO block average "
                "(clean vs current), using the current preprocessing toggles."
            ),
            kind="success",
        )

        def _gray(ax, s, pr, title):
            _order = np.lexsort((s["rsd"], s["wl"]))
            _z = np.clip(robust_z(pr["od"][_order], axis=1), -5, 5)
            ax.imshow(_z, aspect="auto", interpolation="nearest",
                      extent=[s["t"][0], s["t"][-1], _z.shape[0], 0], cmap="gray", vmin=-4, vmax=4)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")

        def _block_long_hbo(s, pr):
            _fs = s["fs"]
            _rel = np.arange(-5, 30, 1 / _fs)
            _mask = (s["group"][:s["n_pair"]] == "long 30-40 mm") & pr["good_pair"]
            _segs = []
            for _o in s["onsets"]:
                _a = int((_o - 5) * _fs)
                _b = _a + _rel.size
                if _a < 0 or _b > pr["dHbO"].shape[1]:
                    continue
                _seg = pr["dHbO"][_mask, _a:_b]
                if _seg.size:
                    _segs.append(_seg - _seg[:, _rel < 0].mean(1, keepdims=True))
            if not _segs:
                return _rel, np.zeros_like(_rel)
            return _rel, 1e6 * np.stack(_segs, 0).mean(0).mean(0)

        _fig = plt.figure(figsize=(13, 8))
        _gs = _fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.4, wspace=0.25)
        _gray(_fig.add_subplot(_gs[0, 0]), sim_clean, proc_clean, "Grayplot - clean reference")
        _gray(_fig.add_subplot(_gs[0, 1]), sim, proc, "Grayplot - current run")

        _ax_g = _fig.add_subplot(_gs[1, 0])
        _ax_g.plot(sim_clean["t"], sim_clean["gvtd"], color="tab:blue", lw=1.1, label="clean")
        _ax_g.plot(sim["t"], sim["gvtd"], color="firebrick", lw=1.1, alpha=0.8, label="current")
        _ax_g.set_title("GVTD: clean vs current")
        _ax_g.set_xlabel("Time (s)")
        _ax_g.set_ylabel("GVTD")
        _ax_g.legend(fontsize=7)

        _ax_b = _fig.add_subplot(_gs[1, 1])
        _rc, _hc = _block_long_hbo(sim_clean, proc_clean)
        _rn, _hn = _block_long_hbo(sim, proc)
        _ax_b.plot(_rc, _hc, color="tab:blue", lw=2.0, label="clean HbO")
        _ax_b.plot(_rn, _hn, color=HBO_C, lw=2.0, alpha=0.85, label="current HbO")
        _ax_b.axvspan(0, 20, color="tab:green", alpha=0.10)
        _ax_b.axhline(0, color="0.6", lw=0.6)
        _ax_b.axvline(0, color="black", lw=0.8)
        _ax_b.set_title("Long-channel HbO: clean vs current")
        _ax_b.set_xlabel("Seconds from onset")
        _ax_b.set_ylabel("Delta HbO (uM)")
        _ax_b.legend(fontsize=7)
        _fig.suptitle("Comparison Mode: identical run with and without the artifacts")
        _out = mo.vstack([_note, _fig])
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Reading the plots, in one pass

        1. **Raw light** - light above floor, below saturation, heartbeat visible?
        2. **QC metrics** - fall-off bands, cardiac coupling, and the temporal-std pruning gate.
        3. **Cap view** - did pruning leave usable spatial coverage?
        4. **OD traces / grayplot** - global events or bad rows?
        5. **GVTD** - motion frames to scrub?
        6. **HbO/HbR block average** - HbO up, HbR down, growing toward longer channels?
        7. **Preprocessing effect** - what band-pass and superficial regression each fix.

        A channel can ace steps 1-2 (clean short channel) and still be a poor
        *cortical* channel. That gap between **light quality** and **neural-response
        quality** is what the layered QC is for.

        ---

        ### What is now physics-based

        - **Forward model:** Farrell (1992) semi-infinite diffuse reflectance sets the
          light fall-off from real tissue absorption/scattering; the **real Piglet
          26-source / 31-detector NeuroDOT pad** (NN 10/22/30/36 mm) produces genuine
          nearest-neighbour separation bands.
        - **Spectroscopy:** HbO/HbR changes are projected to OD with the **modified
          Beer-Lambert law** using real Prahl/omlc extinction coefficients and adult
          DPF values, then **inverted** to recover concentrations (the actual 2x2
          MBLL solve).
        - **Preprocessing:** zero-phase Butterworth band-pass (0.02-1 Hz) and
          short-separation (NN1) regression are the real operations a NeuroDOT-style
          pipeline runs.

        ### Honest limitations (still a teaching tool, not an instrument model)

        - **Diffusion approximation, homogeneous semi-infinite medium.** No Monte
          Carlo / FEM, no layered scalp-skull-brain, no real head mesh. So depth
          sensitivity (`cortical_sens`) is a smooth phenomenological weight, not a
          true photon-measurement-density Jacobian.
        - **Cap is the real piglet pad but flattened to 2D** (spos2/dpos2): no head
          curvature, and there is no image reconstruction (no inverse model to a brain
          volume).
        - **DPF values are adult-head (~6.4 / 5.75).** A piglet head has a shorter
          photon path, so absolute HbO/HbR magnitudes would scale - but the QC plot
          *shapes* (the point of this tool) are unaffected.
        - **MBLL assumes a single homogeneous compartment**, so recovered HbO/HbR mix
          cortical and scalp signal - which is exactly why short-separation regression
          matters here.
        - **Physiology is synthetic** (sinusoids + a canonical HRF), not measured.
          Amplitudes are realistic in scale but not subject-specific.
        - **Block averaging is idealised** (perfect onsets, no trial rejection).

        Use it to train your eye on shapes and failure signatures, then trust real
        data and a real pipeline (NeuroDOT, Homer3, NIRFASTer/Monte Carlo for the
        forward model) for anything quantitative.
        """
    )
    return


if __name__ == "__main__":
    app.run()
