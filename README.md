# Knee MRI anatomy and finding classification

MRI classification for twelve findings, cartilage/meniscus segmentation and
ACL/MCL attachment localization. Developed using Kaggle knee MRI examinations,
report-derived finding labels, reviewed masks and my non-expert attachment-point
annotations. Data and the viewer are not included. The evaluated finding
classifier is available as a separate [checkpoint release](https://github.com/michaelzenkay/knee-public/releases/tag/v4-epoch10).

## Method

The [finding classifier](src/knee_mri/model.py) uses DINOv2-small/14 to identify
relevant regions within each slice, combines information across slices within
each MRI series, and then combines across series to predict twelve findings for
the examination. It receives MRI images and acquisition geometry; the separate
anatomy model's outputs are not classifier inputs.

[DICOM preprocessing](src/knee_mri/preprocessing.py) orders slices physically
and retains rectangular images and all slices. MRI intensities are rescaled using
each volume's intensity percentiles, with arctan for extreme values. An optional
frozen ACL localizer anchors a crop expanded
to retain the tissue envelope. Unreliable anchors retain the full view.
Acquisition-matrix metadata can reduce expanded reconstruction grids; inputs are
padded to multiples of 14 for DINOv2.

The separate [anatomy model](src/knee_mri/anatomy.py) uses adjacent-slice triplets
and a KneeXNet-initialized ResNet-34 U-Net. Four independent sigmoid channels
predict femoral, tibial and patellar cartilage and meniscus. Custom ACL/MCL heads
share its encoder/decoder and predict two endpoint heatmaps plus slice-presence
scores. Each endpoint selects a slice and an in-plane heatmap centroid; it is
not direct XYZ regression. The crop localizer is a separate frozen network.

The demo adapter segments sagittal series at 256 pixels and localizes attachments
at 192 pixels: ACL on sagittal, MCL on coronal with DICOM orientation/laterality.
These anatomy outputs are separate from the classifier's twelve finding scores.

## Evaluated checkpoint and MRNet baseline

The [V4 epoch-10 release](https://github.com/michaelzenkay/knee-public/releases/tag/v4-epoch10)
contains the classifier used for the exploratory MRNet baseline. Download
`knee-v4-epoch10.zip` (91 MB) and verify it against the release's `SHA256SUMS`.
The archive includes `selected.pt`, `dino_config.json`, a model card, aggregate
metrics and plots, and validation/provenance records. All 323 tensors match the
evaluated checkpoint exactly; local training metadata was removed, so the
packaged file has a different checksum. The release was checked with source
commit `20fe84c8a3e43ee5a10f8def6d531766cb308676`.

The baseline used 120 MRNet validation/tuning exams, 360 volumes and 11,312
slices. Original DICOM geometry was unavailable: the adapter assumed 0.625 mm
pixels, 3.5 mm slice steps, fixed plane axes and sequence flags. It retained all
released pixels and slices, used the V4 intensity transform, and bypassed the
localizer, crop and resampling. No training or threshold tuning used this cohort.

| Endpoint | ROC AUC | Average precision | Sensitivity at 0.5 | Specificity at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| ACL tear vs ACL finding score | 0.840 | 0.783 | 0.241 | 0.955 |
| Meniscus tear vs max-head proxy | 0.804 | 0.785 | 0.731 | 0.794 |

The meniscus score is `max(Medial Meniscus, Lateral Meniscus)`, an exploratory
proxy for MRNet's pooled label. There is no general-abnormality output. These are
assumed-geometry validation/tuning results, not Stanford hidden-test results or
clinical validation. The fixed 0.5 ACL threshold missed 41 of 54 positive exams;
geometry sensitivity variants remain unevaluated.

This release contains only the finding classifier. The anatomy and ACL crop
localizer checkpoints are not included, so it does not reproduce the full native
crop pipeline by itself. Images, reports, study labels and per-exam predictions
are excluded. See the release's `MODEL_CARD.md` and `TERMS.md` for input,
validation and licensing limits; the source license does not override dataset
or upstream model terms.

## Requirements

Requires Python 3.11+ and PyTorch.

## Inference

```python
from knee_mri.data import FINDINGS
from knee_mri.inference import load_model, predict
from knee_mri.preprocessing import ExamPreprocessor

device = "cuda"  # Or "cpu".
model = load_model("selected.pt", "dino_config.json", device)
preprocessor = ExamPreprocessor("acl/best_checkpoint.pt", "study_split.csv", device)
rows = [{"StudyInstanceUID": "study-a", "SeriesInstanceUID": "sagittal-a",
         "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "1", "Fat_Suppression": "0"}]
images, details = preprocessor.model_inputs("dicom", "study-a", rows)
scores = dict(zip(FINDINGS, predict(model, images, device)))
```

DICOM files live under `dicom/<study>/<series>/`. Single-frame DICOM requires
pixel spacing, image position and orientation; unsupported or mixed grids fail
explicitly. The localizer's `attachment_predictions.csv` accompanies its
checkpoint and is checked against `study_split.csv`. `ExamPreprocessor()` omits
localization and retains the full field; that changes the classifier input policy.

```python
from knee_mri.data import decode_native
from knee_mri.anatomy_inference import load_anatomy, predict_anatomy

volume = decode_native("dicom/study-a/sagittal-a")
anatomy = load_anatomy("anatomy.pt", device)
result = predict_anatomy(volume["pixels"], anatomy, "Sagittal", device=device)
```

Masks are returned as `[slices, 4, 256, 256]`; endpoint x/y are normalized to
[0, 1] with zero-based slice indices. For MCL, pass `"Coronal"`, `laterality="L"`
or `"R"`, and `orientation=volume["orientation"]`. Only load trusted checkpoints.

## Training and evaluation

The [classifier trainer](src/knee_mri/training.py) starts from generic DINOv2,
trains aggregation heads for two epochs, then unfreezes the encoder with a
two-epoch learning-rate ramp. It uses soft-label BCE, AdamW, accumulation over
eight studies and gradient clipping. Selection uses mean AUC across all twelve
findings; individual AUCs and soft-label BCE are also reported. Development must
contain both classes for each finding.

Prepare a `manifest.json` using [this schema](configs/studies.example.json).
Each series NPZ holds unnormalized cropped float32 `pixels`, `positions_mm`,
`spacing_mm` and `orientation` from `ExamPreprocessor.process`. Preserve study
splits and include patient IDs when examinations repeat. Protocol is
`plane * 4 + fluid_sensitive * 2 + fat_suppression`, with sagittal/coronal/axial
numbered 0/1/2. Paths and SHA-256 hashes identify the prepared inputs.

```python
from knee_mri.training import train_classifier
train_classifier("data/cache", "assets/dinov2-small", "runs/reference", device="cuda")
```

The output contains `best.pt`, `last.pt`, DINO configuration, settings and metrics.
This is a new-run trainer; distributed execution and resume are outside this package.

[Anatomy training](src/knee_mri/anatomy_training.py) provides point-target creation
and a balanced epoch function. Segmentation batches contain `image`, `target`
and `supervised_classes`; only annotated channels receive BCE/Dice loss.
Attachment batches contain one contiguous case window, Gaussian heatmaps and
through-plane presence targets. Their loss combines presence BCE, spatial KL,
slice cross-entropy and endpoint slice/separation errors.

Initialize the model with `initialize_from_kneexnet`, then use AdamW with
`optimizer_parameter_groups(encoder_lr=5e-5, decoder_lr=1e-4, head_lr=3e-4,
weight_decay=.01)`. The demonstrated recipe jointly trains the whole network for
40 epochs, taking 32 steps each for segmentation, ACL and MCL per epoch. Supply
class weights derived from training-mask availability and keep subjects disjoint.
Annotation review and dataset-specific batch assembly remain outside the package.
`evaluate_segmentation` reports Dice for annotated channels; `endpoint_error_mm`
measures attachment error after both points are mapped to the same physical frame.

## Tests and scope

```bash
python -B -m unittest discover -s src/tests -v
```

Three CPU checks cover DICOM geometry, classifier gradients/checkpoint loading
and anatomy outputs/losses. They use the installed package; reinstall after edits.
These are software checks, not accuracy estimates. Scores are not calibrated
clinical risks; masks can overextend or miss anatomy.

[DINOv2](https://github.com/facebookresearch/dinov2) is from Meta; the segmentation initialization is from
[KneeXNet-2.5D](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D).
This source is licensed under [Apache 2.0](LICENSE). Data and upstream model
weights retain their separate terms.
