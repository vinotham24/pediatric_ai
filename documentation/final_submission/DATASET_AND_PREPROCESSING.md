# Dataset and Preprocessing

## Cry dataset

The current feature/metadata table has 1,583 rows: 1,389 tagged with one of eight cry-reason categories and 194 marked unknown. The eight-category distribution and usable counts are:

| Category | Rows | Usable rows | Unique recording keys |
|---|---:|---:|---:|
| belly_pain | 43 | 43 | 27 |
| burping | 25 | 25 | 17 |
| cold_hot | 19 | 19 | 12 |
| discomfort | 58 | 58 | 38 |
| hungry | 1,124 | 1,124 | 742 |
| lonely | 23 | 23 | 15 |
| scared | 15 | 15 | 11 |
| tired | 82 | 80 | 56 |
| **Total modeled categories** | **1,389** | **1,387** | **930** |
| unknown (not a modeled class) | 194 | 194 | — |

There are 1,585 raw files; two malformed `.caf.caf` files are not indexed in the metadata. The 1,583 indexed source formats are 712 3GP, 457 WAV, and 414 CAF. The 457 source recording keys represented in multiple formats are copies/representations, not independent events. One duplicate-content pair is silent. The usable counts exclude two silent tired rows.

Labels are source/contributor suspected reasons derived from documented reason codes; they are not clinically verified. Dataset documentation describes the UUID as an app-installation ID, not a participant ID. Age metadata is an age-group field, not verified individual patient identity. Seven cold_hot-coded files in the cleaned discomfort folder need review. No automatic relabeling was performed.

## Cough dataset

West China originals are kept under `datasets/cough/raw/west_china/`: 82 pneumonia-group and 91 bronchitis-group MP3 files, indexed P1 through P82 and B1 through B91. The paper reports a cohort age range of 0 to 11 years, but no released per-file age mapping is available; event boundaries are absent. The cohort labels are not cough-event annotations. Current annotation/event CSVs are empty (zero rows), and processed cough/non-cough directories have no clips. Consequently, no cough-event class distribution or classifier metric is available.

The official dataset record is https://doi.org/10.6084/m9.figshare.21176197.v1. Cite Liao, Siqi; Song, Chao; Wang, Xiaoqin; Wang, Yanyun (2022), *Data from: A classification framework for identifying bronchitis and pneumonia in children based on a small-scale cough sounds dataset*, Figshare. Apply CC BY 4.0 attribution, link https://creativecommons.org/licenses/by/4.0/, state changes, and do not imply endorsement. No new dataset was downloaded for this work.

## Audio processing

Existing cry preprocessing converts source recordings to mono 16 kHz. Existing feature extraction uses 91 numeric columns and a five-second window; the Phase 6 audit found typical duration near seven seconds, so late audio may be truncated. This issue is documented; this completion did not rewrite features or training audio. Upload/live input is validated and decoded before inference, with empty/corrupt/unsupported files returning errors. Cough segmentation is annotation-driven and cannot produce clips until verified annotations exist.

## Leakage and provenance

Historical splits group app-installation UUIDs and source-recording keys, but not verified patients. This reduces same-key copy leakage but does not establish patient independence. Future experiments must group by participant when valid identifiers become available and keep every copy/segment from a source event in one split. Historical test results are descriptive because test data were reused in prior model work.
