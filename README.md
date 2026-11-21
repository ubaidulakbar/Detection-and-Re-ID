# Stamp Detection and Re-Identification Dataset

This repository provides datasets, organization details, and supporting resources developed for research on document stamp detection and instance-level stamp re-identification (Re-ID).

> **Associated manuscript:** Under review  
> **Detection dataset:** 1,549 images  
> **Re-ID dataset:** 6,690 images

The datasets are associated with an ongoing research study. The corresponding paper is currently under review. Citation details for the paper will be added here once they become publicly available.

---

## Datasets

### 1. YOLO-Stamp Detection Dataset

The detection dataset is designed for localizing and classifying stamps in document images.

- **Total images:** 1,549
- **Classes:** Circle, Triangle, Rectangle
- **Training:** 1,089 images
- **Validation:** 230 images
- **Testing:** 230 images
- **Split:** 70% / 15% / 15%
- **Preprocessing:** None in the published Roboflow version
- **Augmentation:** None in the published Roboflow version

The dataset was constructed from multiple stamp-related datasets, including StaVer, DDI-100, and other publicly available Roboflow projects. Outliers such as seals and logos were removed during dataset preparation.

**Public dataset:**  
https://universe.roboflow.com/stamp-detection-hnojr/yolo-stamp

---

### 2. Stamp Re-Identification Dataset

The Re-ID dataset is designed for metric-learning experiments involving instance-level stamp matching.

Unlike the detection dataset, which focuses on localization, the Re-ID dataset is organized around individual **stamp identities**. Multiple visual variations are provided for each identity so that models can learn identity-consistent representations under changes in ink intensity, resolution, noise, distortion, and partial occlusion.

Because access to large collections of real organizational stamps is restricted by privacy, security, and institutional constraints, this dataset was produced using an **AI-assisted image generation process** to create controlled visual variations without exposing confidential organizational stamps.

### Download

**Re-ID dataset (Google Drive):**  
https://drive.google.com/drive/folders/1Vn9GZzvC53Qt6rvT-FciMLmwhGet_NrC?usp=sharing

#### Dataset size

| Split | Images |
|---|---:|
| Training | 6,063 |
| Testing | 627 |
| **Total** | **6,690** |

#### Training groups

| Visual group | Images |
|---|---:|
| Circular-type | 2,089 |
| Rectangular-type | 2,006 |
| Triangular-type | 1,968 |

#### Testing groups

| Visual group | Images |
|---|---:|
| Circular-type | 217 |
| Rectangular-type | 184 |
| Triangular-type | 226 |

Identity grouping is assisted by feature representations extracted using a pretrained **ResNet50** model. Image-level embeddings are used to form identity-level representations, allowing visually related identities to be grouped systematically while preserving each stamp identity in a separate directory.

This organization supports structured triplet sampling:

- **Anchor:** image from a stamp identity
- **Positive:** another image from the same identity
- **Negative:** image from a different identity

Negatives may be sampled from visually similar identities within the same group or from identities belonging to different visual groups.

Fixed evaluation triplets are used to support consistent and reproducible comparisons across sampling strategies and repeated experiments.

An optional **base group** is also supported for verification scenarios in which an organization has an official reference stamp and wishes to compare it with visually similar, distorted, altered, or potentially forged variants.

---

## Intended Use

These resources are intended primarily for academic and research use in areas such as:

- document stamp detection;
- stamp verification;
- metric learning;
- stamp re-identification;
- document-image analysis;
- similarity learning;
- representation learning; and
- controlled experiments involving visually similar stamp instances.

The Re-ID dataset should not be interpreted as a collection of genuine organizational stamps or confirmed forged stamps. AI-assisted samples are used to support controlled research experiments where access to confidential organizational stamp data is limited.

---

## Reproducibility

The dataset organization is designed to support reproducible metric-learning experiments. In particular:

- training and testing sets are kept separate;
- identities remain separated into identity-specific folders;
- visual grouping is embedding-guided rather than based solely on subjective manual grouping; and
- fixed triplets can be used for consistent evaluation across experiments.

---

## Associated Paper

The manuscript associated with these datasets is currently under review.

**Paper citation:**

```bibtex
% BibTeX will be added after publication.
```

---

## Citation

If you use these datasets, this repository, or the associated code in academic research, please cite the relevant dataset and, once available, the associated paper.

---

## License and Source Data

Please review the licenses of the individual source datasets before redistribution or commercial use.

The detection dataset incorporates material derived from multiple publicly available stamp datasets, including **StaVer**, **DDI-100**, and other Roboflow projects. Users should also cite original datasets where required by their respective licenses and publications.

Licensing of code and other resources in this repository should be considered separately from the licenses of any source datasets.

---

## Contact

For questions regarding the datasets, reproducibility, or research use:

**Ubaid Ul Akbar**  
GitHub: https://github.com/ubaidulakbar  
Portfolio: https://ubaidulakbar.github.io/portfolio/
