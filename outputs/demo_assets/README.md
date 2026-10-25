# Clinical Turing Test Benchmark: Real vs. Synthetic OCT Discrimination

## 1. Regulatory Context & CDSCO Investigational Status

> [!WARNING]
> ### CDSCO Investigational Medical Device Notice
> **Statutory Citation**: Rule 51, Medical Device Rules (MDR) 2017 — Directorate General of Health Services, Central Drugs Standard Control Organization (CDSCO), Ministry of Health & Family Welfare, Government of India.
> 
> - **Regulatory Classification**: Investigational Software as a Medical Device (SaMD) / In-Silico Computational Evaluation Platform.
> - **Intended Use**: Academic research, visual generative Turing testing, algorithmic artifact auditing, and clinical discrimination benchmarking.
> - **Non-Diagnostic Limitation**: Synthetic OCT B-scans and volumetric reconstructions are computer-generated approximations produced by deep generative diffusion models. They are **NOT** approved or certified as clinical diagnostic records or in-vitro/in-vivo medical devices under the Drugs and Cosmetics Act, 1940.
> - **Strict Prohibition**: This module, its benchmark datasets, and generated findings must **NEVER** be used as standalone clinical evidence, for primary ophthalmic triage, or to substitute standard ocular diagnostics by a licensed ophthalmologist.

---

## 2. Benchmark Architecture & Objectives

The **Clinical Turing Test** quantifies whether deep generative models (such as the Latent Volumetric Diffusion Model, LVDM) generate retinal Optical Coherence Tomography (OCT) cross-sections that are perceptually and anatomically indistinguishable from authentic in-vivo OCT B-scans.

### Primary Metrics
1. **Human Reader Accuracy ($\%$ Correct)**:
   $$\text{Accuracy} = \frac{\text{Correct Real Identifications} + \text{Correct Synthetic Identifications}}{\text{Total Evaluated Scans}} \times 100$$
2. **AI Turing Fooling Rate ($\%$)**:
   $$\text{Fooling Rate} = \frac{\text{Synthetic OCT Scans Diagnosed as 'Real'}}{\text{Total Synthetic OCT Scans Evaluated}} \times 100$$
   *A fooling rate approaching $50\%$ indicates true clinical indistinguishability (chance performance by the clinical expert).*
3. **Realism Likert Score (1–5)**:
   - `1`: Obvious generative artifacts, disjointed layers, or non-physiological contours.
   - `2`: Recognizable retina, but noticeable blur or synthetic speckle irregularities.
   - `3`: Plausible anatomical stratification with minor edge inconsistencies.
   - `4`: High-grade clinical resemblance with clear inner/outer retinal distinction.
   - `5`: Indistinguishable from real clinical hardware capture.
4. **Layer Definition & Boundary Sharpness (1–5)**:
   Scores resolution of critical micro-structures:
   - Vitreous optical cavity
   - Inner Limiting Membrane (ILM) & RNFL
   - Inner & Outer Plexiform / Nuclear Layers (IPL, INL, OPL, ONL)
   - Photoreceptor Ellipsoid Zone (IS/OS junction)
   - Retinal Pigment Epithelium (RPE) & Bruch's Membrane complex
   - Choriocapillaris and scleral optical backscatter attenuation

---

## 3. Directory Structure & Key Assets

```
outputs/demo_assets/
├── clinical_workflow_walkthrough.mp4 # 2-minute 720p HD clinical walkthrough demonstration video
├── hf_spaces_demo_qr.png             # Scannable QR code linking directly to live HF Spaces studio
├── generate_qr.py                    # Script generating the branded HF Spaces QR code asset
├── record_walkthrough.py             # Video synthesis script generating the 1200-frame MP4
├── walkthrough_script.md             # Spoken narration script & timestamped chapter storyboard
├── clinical_turing_test.py           # Core evaluation engine, scoring math & session manager
├── README.md                         # Regulatory documentation & clinical testing guide
├── turing_manifest.json              # Trial catalog with blinded IDs and ground truth
├── reader_evaluations.json           # Reader trial logs and session telemetry
└── samples/                          # Curated paired benchmark B-scans
    ├── sample_OCT-TR-R001.png        # Real in-vivo OCT scan
    ├── sample_OCT-TR-R002.png        # Real in-vivo OCT scan
    ...
    ├── sample_OCT-TR-S001.png        # Synthetic LVDM diffusion OCT scan
    ├── sample_OCT-TR-S002.png        # Synthetic LVDM diffusion OCT scan
    ...
```

---

## 4. 2-Minute Demonstration Walkthrough Video & Storyboard

The video asset `clinical_workflow_walkthrough.mp4` demonstrates the complete end-to-end clinical workflow over 120 seconds (1,200 frames @ 10 FPS, 1280x720 HD):

1. **Chapter 1 (0:00 - 0:24)**: *Introduction & CDSCO Regulatory Scope* — Statutory citations under Rule 51 of Medical Device Rules (MDR) 2017, investigational SaMD classification, and system architecture.
2. **Chapter 2 (0:24 - 0:48)**: *Real-Time Auto-Capture & Quality Control* — Tenengrad spectral sharpness, modified Laplacian focus, optic disc & macula anatomical geometry, and 3-frame stability lock-in.
3. **Chapter 3 (0:48 - 1:12)**: *Cross-Modal 3D Volumetric Diffusion* — EyeFound 512D conditioning, 3D latent UNet denoising, and continuous 32-slice coherent OCT volume synthesis.
4. **Chapter 4 (1:12 - 1:36)**: *8-Layer Retinal Segmentation & Multimodal Classifier Uplift* — Sub-micron boundary tracing (ILM to Choroid) and diagnostic confidence boost from 2D fundus baseline to 3D multimodal fusion.
5. **Chapter 5 (1:36 - 2:00)**: *Clinical Turing Discrimination & Live Demo Access* — Blind reader challenge, indistinguishability telemetry, and live Hugging Face Spaces QR code connection.

---

## 5. Live Hugging Face Spaces Access QR Code

- **Live Demo URL**: [https://huggingface.co/spaces/Charan9441/Fund2Oct](https://huggingface.co/spaces/Charan9441/Fund2Oct)
- **High-Resolution QR Asset**: [`outputs/demo_assets/hf_spaces_demo_qr.png`](file:///c:/Users/racha/OneDrive/Desktop/Fund2Oct/Func2Oct/outputs/demo_assets/hf_spaces_demo_qr.png)
- **Generator Script**: [`outputs/demo_assets/generate_qr.py`](file:///c:/Users/racha/OneDrive/Desktop/Fund2Oct/Func2Oct/outputs/demo_assets/generate_qr.py)

---

## 6. Localized Telugu Voice Narration & Accessibility

For rural primary health center (PHC) deployment, the studio integrates text-to-speech accessibility using Google Text-to-Speech (`gTTS`):
- **Live Auto-Capture Voice Prompts**: Real-time Telugu verbal feedback during patient scanning (e.g., *"దృష్టిని అలాగే ఉంచండి. కదలకుండా స్థిరంగా ఉండండి"*).
- **Patient Audio Health Summary**: Automated patient-facing audio report explaining diagnostic results in accessible Telugu with CDSCO non-diagnostic investigational guidance.

---

## 4. Programmatic Usage

```python
from outputs.demo_assets.clinical_turing_test import ClinicalTuringTestSession

# Initialize session
session = ClinicalTuringTestSession()

# Fetch random blind test case
case = session.get_blind_case()
print("Evaluating Case ID:", case["case_id"])
print("Pathology Type:", case["pathology_category"])

# Clinician submits diagnosis
result = session.submit_evaluation(
    case_id=case["case_id"],
    reader_choice="Synthetic OCT",
    confidence_pct=85.0,
    realism_score=4,
    layer_clarity_score=4,
    observed_artifacts=["Subtle choroidal smoothing"],
    reader_notes="Photoreceptor layer looks realistic, but choroidal vessel borders are slightly diffuse."
)

print("Result:", result["verdict_badge"])
print("Summary:", result["verdict_summary"])
print("Session Metrics:", session.get_summary_metrics())
```
