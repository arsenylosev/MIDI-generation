# whole-song-gen GitHub Repository Analysis

## Repository: ZZWaang/whole-song-gen
- **Paper:** Ziyu Wang, Lejun Min, and Gus Xia. "Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models." ICLR 2024.
- **License:** MIT
- **Language:** Python 100%
- **Stars:** 53, Forks: 3

## Repository Structure:
- `data/` - Training data (download links in download_link.txt)
- `data_utils/` - Data utilities
- `experiments/` - Experiment configurations
- `inference/` - Inference code
- `model/` - Model architecture code
- `params/` - Parameter configurations
- `pretrained_models/` - Pretrained VAEs (download links)
- `results_default/` - Cascaded Diffusion Models (download links)
- `train/` - Training code
- `inference_whole_song.py` - Main inference script
- `train_main.py` - Main training script

## Architecture: 4-Level Cascaded Diffusion
1. **Form (frm)** - Generates song structure (phrase types and keys)
2. **Counterpoint (ctp)** - Generates counterpoint/draft level
3. **Lead Sheet (lsh)** - Generates lead sheet (melody + chords)
4. **Accompaniment (acc)** - Generates full accompaniment

## Training Commands:
```
python train_main.py --mode frm  # Form level
python train_main.py --mode ctp --autoreg --mask_bg  # Counterpoint
python train_main.py --mode lsh --autoreg --mask_bg  # Lead sheet
python train_main.py --mode acc --autoreg --mask_bg  # Accompaniment
```

## Inference:
```
# With specified form (e.g., intro-4bars, A-4bars, A-4bars, B-8bars, bridge-4bars, A-4bars, B-8bars, outro-4bars, key=G major)
python inference_whole_song.py --nsample 4 --pstring i4A4A4B8b4A4B8o4 --key 7

# With generated form
python inference_whole_song.py --nsample 4
```

## Key Input Format:
- Form string: `i4A4A4B8b4A4B8o4` (section types + bar counts)
- Key: integer (e.g., 7 = G major)
- Output: MIDI files in `demo/` directory

## Data: POP909 dataset (curated pop songs)
- Piano-roll representation (image-like)
- Each level has its own diffusion model

## Detailed Code Analysis

### Data Representation (McpaMusic class):
- **MCPA** = Melody, Chords, Phrase, Accompaniment
- Uses `num_beat_per_measure=4`, `num_step_per_beat=4` (16th note resolution)
- Piano roll format: (2, L, 128) - onset and sustain channels, 128 MIDI pitches
- Melody boundary at pitch 48 (C3) - below is chord, above is melody

### MIDI Output:
- Uses `pretty_midi` library
- Converts piano roll → note matrix → MIDI notes
- Output: 5 tracks (red_mel, red_chd, mel, chd, acc) or 3 tracks
- Default BPM: 90, unit: 0.25 (quarter note)

### Model Parameters:
- **Form level:** in_channels=8, out_channels=8, channels=64, 1000 diffusion steps
- **Counterpoint:** in_channels=10, out_channels=2, channels=64
- **Lead Sheet:** similar architecture
- **Accompaniment:** similar architecture
- All use U-Net with attention at levels [2,3], channel_multipliers=[1,2,4,4]
- Latent diffusion with scaling factor 0.18215

### Key Interface Points (for our MIDI generation module):
1. **Input to the pipeline:** Form string (e.g., "i4A4A4B8b4A4B8o4") + key (0-11) + major/minor
2. **Output from the pipeline:** MIDI files with multiple tracks
3. **Internal representation:** Piano roll (2, L, 128) numpy arrays
4. **The pipeline is sequential:** Form → Counterpoint → Lead Sheet → Accompaniment
