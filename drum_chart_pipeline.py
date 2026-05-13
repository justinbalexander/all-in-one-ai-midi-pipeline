#!/usr/bin/env -S uv run --python 3.10
"""End-to-end Clone Hero drum chart generation pipeline.

Separates drums from audio, detects onsets, classifies with DrumSep
sub-stem transient analysis, and outputs a .chart file with proper
cymbal lane mapping (ride=blue cymbal, hi-hat=yellow cymbal, etc.).

Usage:
    # With venv activated:
    source .venv/bin/activate
    python drum_chart_pipeline.py --input song.mp3 --output ./chart-folder/

    # Or with uv run (uses .venv in project or creates one):
    uv run drum_chart_pipeline.py --input song.mp3 --output ./chart-folder/

    # Keep intermediate files for tuning:
    python drum_chart_pipeline.py --input song.mp3 --output ./chart/ --keep-temp

Setup:
    1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
    2. Clone this repo (all-in-one-ai-midi-pipeline)
    3. Clone the MSST repo as a sibling directory:
         cd .. && git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git
    4. Install dependencies into the venv (uv auto-creates it with Python 3.10):
         uv pip install --python 3.10 demucs adtof_pytorch pretty_midi soundfile numpy torch
         uv pip install --python 3.10 wandb ml_collections loralib timm==0.9.2 einops librosa
    5. Download the DrumSep v0.1 model into this repo's root directory:
         https://github.com/jarredou/models/releases/tag/aufr33-jarredou_MDX23C_DrumSep_model_v0.1
       You need both the .ckpt and .yaml files.

Expected directory layout:
    parent_dir/
      all-in-one-ai-midi-pipeline/       <-- this repo (script lives here)
        drum_chart_pipeline.py
        .venv/                           <-- Python 3.10 venv
        aufr33-jarredou_..._model.ckpt   <-- downloaded model
        aufr33-jarredou_..._model.yaml
      Music-Source-Separation-Training/  <-- MSST repo (sibling)

Output folder contains:
    notes.chart    -- Clone Hero drum chart (7-class: kick/snare/toms/hh/ride/crash)
    song.mp3       -- copy of the input audio


Tuning the classification thresholds
=====================================

The pipeline has 5 steps with very different runtimes:

    Step 1: Demucs separation        ~3 min   (CPU) / ~30s (GPU)
    Step 2: ADTOF transcription      ~13s
    Step 3: DrumSep sub-stem sep     ~30 min  (CPU) / ~1-2 min (GPU)
    Step 4: Classification           ~0.6s    <--- THIS is what you tune
    Step 5: Chart generation         ~0.1s

Steps 1-3 are the bottleneck but only need to run ONCE per song. Their
outputs are cached in the temp directory. Step 4 (classification) is
nearly instant, so you can iterate on threshold tuning in sub-second
loops against cached data.

Quick tuning workflow for a single song:

    1. Run the full pipeline once with --keep-temp:
         python drum_chart_pipeline.py --input song.mp3 --output ./chart/ --keep-temp

    2. Note the temp directory path printed in step output. It contains:
         tmpXXX/
           htdemucs_6s/<song>/drums.wav      <-- drums stem from Demucs
           adtof_drums.mid                    <-- onset MIDI from ADTOF
           drumsep_input/drums.wav            <-- copy of drums stem
           drumsep_output/drums/*.wav         <-- 6 sub-stems from DrumSep

    3. Write a tuning script that reuses the cached files:
         - Load drumsep_output/drums/*.wav (the 6 sub-stems)
         - Load adtof_drums.mid (the onset times)
         - Call _classify_one() with different threshold values
         - Compare results against a manual .chart file
         - Sweep CYMBAL_RESCUE_THRESHOLD (currently 0.15) and
           RIDE_OVERRIDE_RATIO (currently 0.5)

    4. For bulk tuning across many songs with a GPU:
         - Rent a cloud GPU instance (any Nvidia CUDA GPU works)
         - Run the full pipeline on each song (Steps 1-3 take ~2 min on GPU)
         - Cache all intermediate outputs
         - Then run the fast classification sweep across the whole dataset

Comparing against manual charts:

    The manual chart is a .chart file with [ExpertDrums] section. Parse
    it to extract (tick, lane, cymbal_toggle) tuples. Map back to drum
    classes using the inverse of CH_MAP. Then compute precision/recall
    per class against the auto-chart output.

Ideal songs for tuning:

    The classification heuristic struggles most with:
    - Crash vs ride disambiguation (similar-sounding cymbals)
    - Cymbal rescue (ADTOF says cymbal but DrumSep picks kick/snare)
    - Simultaneous hits (kick + ride, kick + crash)

    Pick songs that feature:
    - Verses with ride cymbal + kick (tests crash/ride separation)
    - Choruses with crash + kick (tests crash detection)
    - Hi-hat grooves with ghost snare (tests hh/snare distinction)
    - Frequent kick + cymbal simultaneous hits (tests cymbal rescue)

    Avoid songs with heavy toms, double kick, or blast beats for initial
    tuning -- those confuse all three models (ADTOF, DrumSep, and the
    heuristic). Start with straightforward rock/pop-rock drumming.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths resolved relative to this script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_CKPT = SCRIPT_DIR / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt"
MODEL_YAML = SCRIPT_DIR / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
MSST_REPO = SCRIPT_DIR.parent / "Music-Source-Separation-Training"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESOLUTION = 192
DRUM_CLASSES = ["kick", "snare", "toms", "hh", "ride", "crash"]

# Clone Hero mapping: drum class -> (lane_number, cymbal_toggle_or_None)
# Lane: 0=kick, 1=red(snare), 2=yellow, 3=blue, 4=green
# Cymbal toggles: 66=yellow cymbal, 67=blue cymbal, 68=green cymbal
CH_MAP = {
    "kick":  (0, None),
    "snare": (1, None),
    "toms":  (4, None),
    "hh":    (2, 66),
    "ride":  (3, 67),
    "crash": (4, 68),
}

# ADTOF MIDI pitch -> broad class hint
ADTOF_PITCH_CLASS = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare",
    42: "hh", 44: "hh", 46: "hh",
    47: "toms", 48: "toms",
    49: "crash", 51: "crash",
}


# ---------------------------------------------------------------------------
# Step 1: Demucs separation
# ---------------------------------------------------------------------------
def run_demucs(input_audio, temp_dir):
    """Run htdemucs_6s to extract the drums stem. Returns path to drums.wav."""
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs_6s",
        "-o", str(temp_dir),
        str(input_audio),
    ]
    subprocess.run(cmd, check=True)

    song_name = Path(input_audio).stem
    drums_path = Path(temp_dir) / "htdemucs_6s" / song_name / "drums.wav"
    if not drums_path.exists():
        raise RuntimeError(f"Demucs did not produce drums stem at {drums_path}")
    return drums_path


# ---------------------------------------------------------------------------
# Step 2: ADTOF drum transcription
# ---------------------------------------------------------------------------
def run_adtof(drums_stem_path, temp_dir):
    """Run ADTOF on the drums stem to get onset MIDI. Returns MIDI path."""
    from adtof_pytorch import transcribe_to_midi

    midi_out = Path(temp_dir) / "adtof_drums.mid"
    transcribe_to_midi(str(drums_stem_path), str(midi_out), device="cpu")
    if not midi_out.exists():
        raise RuntimeError("ADTOF did not produce output MIDI")
    return str(midi_out)


# ---------------------------------------------------------------------------
# Step 3: DrumSep sub-stem separation
# ---------------------------------------------------------------------------
def run_drumsep(drums_stem_path, temp_dir):
    """Run DrumSep to separate drums into 6 sub-stems. Returns {name: audio_array}."""
    # Insert MSST repo so we can import proc_folder
    msst_str = str(MSST_REPO)
    if msst_str not in sys.path:
        sys.path.insert(0, msst_str)
    from inference import proc_folder

    # DrumSep needs a folder input - put drums.wav in a dedicated dir
    input_folder = Path(temp_dir) / "drumsep_input"
    input_folder.mkdir(exist_ok=True)
    shutil.copy2(drums_stem_path, input_folder / "drums.wav")

    output_folder = Path(temp_dir) / "drumsep_output"

    proc_folder({
        "model_type": "mdx23c",
        "config_path": str(MODEL_YAML),
        "start_check_point": str(MODEL_CKPT),
        "input_folder": str(input_folder),
        "store_dir": str(output_folder),
        "filename_template": "{file_name}/{instr}",
        "device_ids": [0],
        "force_cpu": not _cuda_available(),
        "use_tta": False,
        "bigshifts": 1,
        "draw_spectro": 0,
        "extract_instrumental": False,
        "disable_detailed_pbar": False,
        "flac_file": False,
        "pcm_type": "FLOAT",
    })

    # Load the 6 sub-stems into memory
    stems = {}
    sr = 44100
    for name in DRUM_CLASSES:
        path = output_folder / "drums" / f"{name}.wav"
        if not path.exists():
            raise RuntimeError(f"DrumSep did not produce {name} stem at {path}")
        audio, file_sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        stems[name] = audio.astype(np.float32)
        sr = file_sr
    return stems, sr


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Step 4: Transient-based onset classification
#
# This is the core heuristic of the pipeline. It combines two AI models:
#
#   - ADTOF: detects WHEN drum hits happen (onset times) and classifies
#     them into 5 broad categories (kick=35/36, snare=38, hihat=42/44/46,
#     tom=47/48, crash=49/51). ADTOF cannot distinguish ride from crash.
#
#   - DrumSep (Jarredou MDX23C v0.1): separates the drum stem into 6
#     sub-stem audio files (kick, snare, toms, hh, ride, crash). Each
#     sub-stem is an isolated audio track containing only that instrument.
#
# The classification strategy measures TRANSIENT ENERGY in each DrumSep
# sub-stem at each ADTOF onset position. The stem with the largest
# transient (energy spike at the hit moment) is presumed to be the
# instrument that was actually struck.
#
# ## Why transient energy, not raw energy?
#
# DrumSep's sub-stems are not perfectly isolated -- there is bleed/leakage
# between them. A kick hit produces a loud thump that bleeds into every
# other stem. If you just measure raw RMS energy at an onset, kick almost
# always wins because it has the most low-frequency energy. By measuring
# the TRANSIENT (RMS at onset minus RMS just before it), we capture the
# *change* in energy, which is more specific to the instrument that was
# actually hit at that moment.
#
# ## The "cymbal rescue" heuristic
#
# Even with transient detection, kick and snare still dominate because
# their transients are sharper and louder than cymbal transients. In early
# testing on "Half Measure" by 1982, raw transient classification produced
# only 3 hi-hats and 0 crashes -- clearly wrong for a rock song.
#
# The rescue logic works as follows: if ADTOF classified a hit as a
# cymbal-type (hi-hat or crash pitch) but DrumSep's transient picked kick
# or snare, we check whether ANY cymbal stem has a transient above 15% of
# the winner's transient. If yes, we override to that cymbal.
#
# The reasoning: ADTOF is specifically trained on drum transcription and
# is reasonably good at distinguishing cymbal-type hits from kick/snare,
# even if it can't differentiate crash from ride. If ADTOF says "this is
# a cymbal" and DrumSep shows even modest cymbal activity, the hit is
# more likely a cymbal than a kick. The 15% threshold is a guess chosen
# to produce plausible-looking results -- it has NOT been validated
# against ground truth.
#
# ## Crash vs ride disambiguation
#
# DrumSep separates crash and ride into different stems, but the
# separation is imperfect -- crash and ride often bleed into each other.
# When crash wins the transient comparison, we check if ride has at least
# 50% of crash's transient. If so, we switch to ride. The rationale is
# that ride cymbals produce a more sustained, less transient-heavy sound
# than crashes, so ride's transient will naturally be lower even when it
# was the actual instrument. The 50% threshold is also unvalidated.
#
# ## Known weaknesses
#
# 1. ALL thresholds (15% cymbal rescue, 50% ride disambiguation, attack
#    window of 512 samples, pre-onset window of 1024 samples) are guesses.
#    They produced reasonable results on one song. They need to be tuned
#    against a dataset of manually-charted songs with known ground truth.
#
# 2. The rescue logic creates a circular dependency: we trust ADTOF to
#    correct DrumSep, but ADTOF itself has errors. A kick misclassified
#    by ADTOF as "crash" could get rescued to a cymbal, producing a false
#    positive. Conversely, a cymbal misclassified by ADTOF as "tom" will
#    never get rescued, producing a false negative.
#
# 3. The transient energy metric is simplistic. It only looks at RMS in
#    fixed windows. A more sophisticated approach could use spectral
#    features (cymbals have more high-frequency energy), onset detection
#    functions, or train a small classifier on the DrumSep sub-stem
#    activations directly.
#
# ## Future work
#
# The right way to tune this is to run the pipeline on songs that already
# have manually-charted Clone Hero drum tracks (which the author has in
# large quantities). For each song, compare the auto-chart classification
# against the manual chart and compute precision/recall per drum class.
# Then sweep the thresholds to maximize F1 score. This requires a GPU
# (cloud rental recommended) since DrumSep is the bottleneck at ~30 min
# per song on CPU.
#
# A more ambitious approach: replace the heuristic entirely with a small
# neural network that takes the 6 DrumSep sub-stem activations at each
# ADTOF onset as input and outputs a 7-class classification. Train it on
# the manual chart dataset. This would likely outperform any hand-tuned
# thresholds.
# ---------------------------------------------------------------------------
def _get_transient_energy(stem_audio, onset_sample, attack_len=512, pre_len=1024):
    """Measure transient energy: RMS at onset minus RMS just before it.

    attack_len: samples after onset to measure hit energy (~11.6ms at 44.1kHz)
    pre_len:    samples before onset to measure ambient energy (~23.2ms at 44.1kHz)
    """
    pre_start = max(0, onset_sample - pre_len)
    pre_seg = stem_audio[pre_start:onset_sample]
    pre_rms = np.sqrt(np.mean(pre_seg ** 2)) if len(pre_seg) > 0 else 0.0

    att_seg = stem_audio[onset_sample:onset_sample + attack_len]
    att_rms = np.sqrt(np.mean(att_seg ** 2)) if len(att_seg) > 0 else 0.0

    return max(0.0, att_rms - pre_rms)


def _classify_one(stems, onset_sample, original_pitch=None):
    """Classify a single onset by transient energy across DrumSep stems.

    See the block comment above for full explanation of the heuristic.
    """
    adtof_class = ADTOF_PITCH_CLASS.get(original_pitch) if original_pitch else None

    transients = {name: _get_transient_energy(audio, onset_sample)
                  for name, audio in stems.items()}
    max_t = max(transients.values()) if transients else 0
    if max_t == 0:
        return adtof_class or "kick"

    # Step 1: Pick the stem with the strongest transient
    best = max(transients, key=transients.get)

    # Step 2: Cymbal rescue -- if ADTOF said cymbal but transient picked
    # kick/snare, check if any cymbal stem has meaningful transient activity.
    # CYMBAL_RESCUE_THRESHOLD (0.15) is unvalidated -- needs tuning.
    CYMBAL_RESCUE_THRESHOLD = 0.15
    if adtof_class in ("hh", "crash") and best in ("kick", "snare"):
        cym = {k: v for k, v in transients.items() if k in ("hh", "ride", "crash")}
        best_cym = max(cym, key=cym.get)
        if cym[best_cym] > max_t * CYMBAL_RESCUE_THRESHOLD:
            best = best_cym

    # Step 3: Crash vs ride disambiguation -- if crash won but ride has
    # significant transient too, prefer ride (it's more common in rock).
    # RIDE_OVERRIDE_RATIO (0.5) is unvalidated -- needs tuning.
    RIDE_OVERRIDE_RATIO = 0.5
    if best in ("crash", "ride"):
        if best == "crash" and transients.get("ride", 0) > transients.get("crash", 0) * RIDE_OVERRIDE_RATIO:
            best = "ride"

    return best


def classify_onsets(adtof_midi_path, stems, sr):
    """Classify every ADTOF onset using DrumSep stem transients.

    Returns (onsets_list, bpm_events, bpm) where each onset is a dict
    with keys: time, original_pitch, class.
    """
    midi = pretty_midi.PrettyMIDI(adtof_midi_path)

    tempo_times, tempo_bpms = midi.get_tempo_changes()
    bpm_events = list(zip(tempo_times, tempo_bpms))
    if not bpm_events:
        bpm_events = [(0.0, 120.0)]
    bpm = bpm_events[0][1]

    onsets = []
    for inst in midi.instruments:
        for note in inst.notes:
            onsets.append({
                "time": note.start,
                "sample": int(note.start * sr),
                "original_pitch": note.pitch,
            })

    for onset in onsets:
        onset["class"] = _classify_one(stems, onset["sample"], onset["original_pitch"])

    return onsets, bpm_events, bpm


# ---------------------------------------------------------------------------
# Step 5: Chart generation
# ---------------------------------------------------------------------------
def _seconds_to_ticks(seconds, bpm_events, resolution=RESOLUTION):
    ticks = 0
    prev_time = 0.0
    prev_bpm = bpm_events[0][1]

    for event_time, bpm in bpm_events:
        if event_time >= seconds:
            break
        ticks += int((event_time - prev_time) * prev_bpm / 60 * resolution)
        prev_time = event_time
        prev_bpm = bpm

    ticks += int((seconds - prev_time) * prev_bpm / 60 * resolution)
    return ticks


def generate_chart(onsets, bpm_events, song_name, output_dir):
    """Write notes.chart to output_dir."""
    bpm = bpm_events[0][1]
    ch_bpm = int(round(bpm * 1000))

    # Collect notes by tick, deduplicating simultaneous hits on same lane
    notes_by_tick = {}
    for onset in onsets:
        tick = _seconds_to_ticks(onset["time"], bpm_events)
        lane, cymbal = CH_MAP[onset["class"]]
        notes_by_tick.setdefault(tick, set()).add((lane, cymbal))

    lines = [
        "[Song]", "{",
        f'  Name = "{song_name}"',
        '  Artist = ""',
        '  Charter = "ACE+DrumSep"',
        "  Offset = 0",
        f"  Resolution = {RESOLUTION}",
        '  Player2 = "bass"',
        "  Difficulty = 0",
        "  PreviewStart = 0",
        "  PreviewEnd = 0",
        '  Genre = "Rock"',
        '  MediaType = "cd"',
        '  MusicStream = "song.mp3"',
        "}", "",
        "[SyncTrack]", "{",
        "  0 = TS 4",
        f"  0 = B {ch_bpm}",
        "}", "",
        "[Events]", "{", "}", "",
        "[ExpertDrums]", "{",
    ]

    for tick in sorted(notes_by_tick):
        for lane, cymbal in sorted(notes_by_tick[tick]):
            lines.append(f"  {tick} = N {lane} 0")
            if cymbal is not None:
                lines.append(f"  {tick} = N {cymbal} 0")

    lines.append("}")

    chart_path = Path(output_dir) / "notes.chart"
    chart_path.write_text("\n".join(lines) + "\n")
    return chart_path


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def print_stats(onsets, bpm, duration, output_dir):
    counts = {cls: 0 for cls in DRUM_CLASSES}
    for o in onsets:
        counts[o["class"]] += 1
    total = sum(counts.values())

    # Build lane description strings
    def lane_str(cls):
        lane, cymbal = CH_MAP[cls]
        s = f"N {lane}"
        if cymbal is not None:
            s += f" + N {cymbal}"
        return s

    # Figure out column widths
    w_drum = max(len("Drum"), *(len(c.title()) for c in DRUM_CLASSES))
    w_count = max(len(str(total)), *(len(str(v)) for v in counts.values()))
    w_count = max(w_count, len("Count"))

    w = max(w_drum, 8)
    cw = max(w_count, 5)

    divider = f"  {'-' * (w + 2)}-{ '-' * (cw + 2)}-{ '-' * 20}"
    header = f"  {'Drum':<{w}} | {'Count':>{cw}} | Chart Notes"
    sep = f"  {'-' * (w + 2)}-+-{ '-' * (cw + 2)}-+-{'-' * 20}"

    print()
    print("=" * 50)
    print("  Clone Hero Drum Chart — Generation Complete")
    print("=" * 50)
    print(f"  Duration      : {duration:.1f}s")
    print(f"  BPM           : {bpm:.1f}")
    print(f"  Total Notes   : {total}")
    print()
    print(header)
    print(sep)
    for cls in DRUM_CLASSES:
        c = counts[cls]
        desc = lane_str(cls)
        print(f"  {cls.title():<{w}} | {c:>{cw}} | {desc}")
    print(divider)
    print()
    print(f"  Output: {output_dir}")
    print(f"    notes.chart")
    print(f"    song.mp3")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Clone Hero drum chart generation",
    )
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Output folder to create")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    if not MODEL_CKPT.exists():
        print(f"Error: DrumSep model checkpoint not found: {MODEL_CKPT}")
        print("Download it from: https://github.com/jarredou/models/releases/tag/aufr33-jarredou_MDX23C_DrumSep_model_v0.1")
        sys.exit(1)

    if not MODEL_YAML.exists():
        print(f"Error: DrumSep model config not found: {MODEL_YAML}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    song_name = input_path.stem

    # Temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Step 1: Demucs
        t0 = time.time()
        print(f"[1/5] Running Demucs separation (htdemucs_6s)...")
        drums_path = run_demucs(input_path, temp_dir)
        print(f"      -> drums stem: {drums_path} ({time.time() - t0:.1f}s)")

        # Step 2: ADTOF
        t0 = time.time()
        print(f"[2/5] Running ADTOF drum transcription...")
        midi_path = run_adtof(drums_path, temp_dir)
        pm = pretty_midi.PrettyMIDI(midi_path)
        n_notes = sum(len(inst.notes) for inst in pm.instruments)
        print(f"      -> {n_notes} onsets detected ({time.time() - t0:.1f}s)")

        # Step 3: DrumSep
        t0 = time.time()
        print(f"[3/5] Running DrumSep sub-stem separation (6 stems)...")
        stems, sr = run_drumsep(drums_path, temp_dir)
        print(f"      -> {', '.join(DRUM_CLASSES)} separated ({time.time() - t0:.1f}s)")

        # Step 4: Classify
        t0 = time.time()
        print(f"[4/5] Classifying onsets with transient analysis...")
        onsets, bpm_events, bpm = classify_onsets(midi_path, stems, sr)
        print(f"      -> {len(onsets)} onsets classified ({time.time() - t0:.1f}s)")

        # Step 5: Generate chart
        t0 = time.time()
        print(f"[5/5] Generating Clone Hero chart...")
        chart_path = generate_chart(onsets, bpm_events, song_name, output_dir)
        shutil.copy2(input_path, output_dir / "song.mp3")
        print(f"      -> {chart_path} ({time.time() - t0:.1f}s)")

        # Stats
        duration = pm.get_end_time()
        print_stats(onsets, bpm, duration, output_dir)

    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
