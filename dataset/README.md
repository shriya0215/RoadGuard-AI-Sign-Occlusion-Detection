# German Traffic Sign Recognition Benchmark (GTSRB) & Synthetic Occlusion Dataset

## Overview
The dataset subsystem powers **RoadGuard AI**, an intelligent road sign occlusion detection system designed for autonomous vehicle perception research.

The project operates on two concurrent visual classification objectives:
1. **Traffic Sign Classification:** 43 standardized German road sign classes according to the German Road Traffic Act (StVO).
2. **Binary Occlusion Detection:** Identifies whether a road sign is `Clear` (Class 0) or `Occluded` (Class 1) by physical environmental artifacts (Leaves, Mud, or Graffiti).

---

## GTSRB Official Benchmark Information
- **Original Source:** Institut für Neuroinformatik, Ruhr-Universität Bochum
- **Benchmark Link:** [https://benchmark.ini.rub.de/](https://benchmark.ini.rub.de/)
- **Total Classes:** 43 traffic sign classes
- **Original Resolution:** Various (from 15x15 to 250x250 pixels, normalized in RoadGuard AI to 64x64 RGB)
- **Color Space:** Standard 3-channel RGB

### How to Download and Place Official GTSRB Files
If you wish to train on the raw 300MB+ official benchmark archive:
1. Download `GTSRB_Final_Training_Images.zip` from the official benchmark repository:
   - Official direct URL: `https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip`
   - Or from HuggingFace: `huggingface.co/datasets/bazyl/GTSRB`
2. Extract the archive into the `dataset/` directory. The structure should match:
   ```
   dataset/
   └── GTSRB/
       └── Final_Training/
           └── Images/
               ├── 00000/
               ├── 00001/
               └── ... / 00042/
   ```
   Or simplified:
   ```
   dataset/
   └── Train/
       ├── 0/
       ├── 1/
       └── ... / 42/
   ```
3. Run `python training/train.py`. The data loader will automatically detect your downloaded images!

---

## Synthetic Occlusion Generation (Data Augmentation)
To train the occlusion detection head without data leakage:
1. **Strict Train/Val/Test Split:** Clean images are split first into 70% Train, 15% Validation, and 15% Test.
2. **Independent Occlusion Augmentation:** Synthetic occlusions are generated independently within each split:
   - **Leaves (Foliage):** Simulates tree branches and organic leaf clusters with natural olive/green pigments, veins, and varying opacity.
   - **Mud / Dirt Splatter:** Simulates road spray, clay blotches, and textured grime with realistic splatter radius and noise.
   - **Graffiti / Vandalism:** Simulates spray paint strokes, tag scribbles, and aerosol dispersion with high-contrast color palettes.
3. **Occlusion Intensity:** Scaled continuously from 10% to 95% surface area coverage.
4. **Binary Labels:**
   - `0 = Clear Sign`
   - `1 = Occluded Sign`
