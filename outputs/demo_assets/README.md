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

## 3. Directory Structure

```
outputs/demo_assets/
├── clinical_turing_test.py      # Core evaluation engine, scoring math & session manager
├── README.md                    # Regulatory documentation & clinical testing guide
├── turing_manifest.json         # Trial catalog with blinded IDs and ground truth
├── reader_evaluations.json      # Reader trial logs and session telemetry
└── samples/                     # Curated paired benchmark B-scans
    ├── sample_OCT-TR-R001.png   # Real in-vivo OCT scan
    ├── sample_OCT-TR-R002.png   # Real in-vivo OCT scan
    ...
    ├── sample_OCT-TR-S001.png   # Synthetic LVDM diffusion OCT scan
    ├── sample_OCT-TR-S002.png   # Synthetic LVDM diffusion OCT scan
    ...
```

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
