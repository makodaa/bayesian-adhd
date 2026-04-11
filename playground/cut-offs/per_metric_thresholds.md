# Per-metric cutoff thresholds (precision target 0.85)

Each cutoff is computed one-vs-rest on the full dataset.
A metric flags a subtype when it meets the direction and cutoff (e.g., `rel_beta <= 0.0130`).
Use the **Recommended** cutoffs to label a recording as likely that subtype (precision >= 0.85).

## Recommended cutoffs (precision target met)

### ADHD-C

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_alpha | >= | 0.223973 | 1.000 | 0.011 | 1.000 |
| rel_beta | <= | 0.013036 | 1.000 | 0.011 | 1.000 |
| tar | <= | 0.618060 | 1.000 | 0.011 | 1.000 |
| atr | >= | 1.617967 | 1.000 | 0.011 | 1.000 |

### ADHD-H

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_alpha | <= | 0.012194 | 1.000 | 0.008 | 1.000 |
| rel_beta | >= | 0.199041 | 0.857 | 0.100 | 0.995 |
| tbr | <= | 0.493945 | 0.867 | 0.108 | 0.995 |

### ADHD-I

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| tar | >= | 3.719410 | 0.875 | 0.087 | 0.995 |
| atr | <= | 0.268860 | 0.875 | 0.087 | 0.995 |

### Control

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_theta | <= | 0.039379 | 1.000 | 0.010 | 1.000 |
| tbr | >= | 9.239005 | 1.000 | 0.005 | 1.000 |

## Below-target cutoffs (precision target not met)

### ADHD-C

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_theta | >= | 0.138607 | 0.273 | 0.398 | 0.803 |
| tbr | <= | 2.012752 | 0.182 | 0.614 | 0.486 |

### ADHD-H

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_theta | <= | 0.046344 | 0.333 | 0.025 | 0.986 |
| tar | <= | 0.838757 | 0.417 | 0.042 | 0.984 |
| atr | >= | 1.192241 | 0.417 | 0.042 | 0.984 |

### ADHD-I

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_theta | >= | 0.235306 | 0.545 | 0.037 | 0.987 |
| rel_alpha | <= | 0.017495 | 0.700 | 0.044 | 0.992 |
| rel_beta | <= | 0.016252 | 0.667 | 0.050 | 0.990 |
| tbr | >= | 4.338676 | 0.667 | 0.100 | 0.980 |

### Control

| Metric | Direction | Cutoff | Precision | Recall | Specificity |
| --- | --- | --- | --- | --- | --- |
| rel_alpha | <= | 0.014764 | 0.667 | 0.010 | 0.997 |
| rel_beta | <= | 0.013957 | 0.667 | 0.010 | 0.997 |
| tar | <= | 0.650019 | 0.500 | 0.005 | 0.997 |
| atr | >= | 1.538416 | 0.500 | 0.005 | 0.997 |
