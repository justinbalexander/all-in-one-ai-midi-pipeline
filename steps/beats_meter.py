import librosa
import numpy as np

DEFAULT_SR = 44100


def _normalize_tempo(bpm: float) -> float:
    """
    Normalize beat tracker tempo by correcting half-time / double-time confusions.
    Keeps the tempo as close to the detected value as possible while staying in
    a plausible range (70–180 BPM).  Prefers the candidate nearest to the raw
    detection so that the actual musical feel is preserved.
    """
    if bpm <= 0:
        return 120.0

    base = float(bpm)

    # Only correct obvious half/double-time errors; avoid aggressive re-mapping.
    ratios = [1.0, 0.5, 2.0]

    raw_candidates = [base * r for r in ratios]
    candidates = [c for c in raw_candidates if 70.0 <= c <= 180.0]
    if not candidates:
        candidates = [base]

    # Pick the candidate closest to the original detection (preserves musical feel).
    best = min(candidates, key=lambda t: abs(t - base))
    return float(best)


def estimate_tempo_downbeats_meter(stems, CFG, manifest):
    """
    Estimate global tempo & downbeats from the original mix.

    Returns:
        {
          "tempo": float,
          "downbeats": [times in seconds],
          "meter": {"numerator": 4, "denominator": 4, "confidence": float},
          "time_signature_written": False
        }

    Also sets manifest["meter_key"]["tempo"] to the normalized tempo.
    """
    # Prefer original source audio
    audio_path = manifest.get("source_audio")

    # Fallback: any available stem
    if not audio_path:
        for v in stems.values():
            if isinstance(v, str):
                audio_path = v
                break

    if not audio_path:
        tempo = 120.0
        info = {
            "tempo": tempo,
            "downbeats": [],
            "meter": {"numerator": 4, "denominator": 4, "confidence": 0.0},
            "time_signature_written": False,
        }
        manifest.setdefault("meter_key", {})["tempo"] = tempo
        print("[beats_meter] No audio found, defaulting tempo=120.0")
        return info

    # Load audio
    sr = CFG.get("sample_rate", DEFAULT_SR)
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Beat tracking in frames
    raw_tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    norm_tempo = _normalize_tempo(float(raw_tempo))

    # Beats -> times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Naive 4/4: every 4th beat is a downbeat
    if len(beat_times) >= 4:
        downbeats = beat_times[::4].tolist()
        confidence = 0.9
    else:
        downbeats = []
        confidence = 0.1

    info = {
        "tempo": norm_tempo,
        "downbeats": [float(t) for t in downbeats],
        "meter": {
            "numerator": 4,
            "denominator": 4,
            "confidence": float(confidence),
        },
        "time_signature_written": False,
    }

    mk = manifest.setdefault("meter_key", {})
    mk["tempo"] = float(norm_tempo)

    print(
        f"[beats_meter] raw_tempo={raw_tempo:.3f}, "
        f"normalized={norm_tempo:.3f}, "
        f"downbeats={len(downbeats)}"
    )

    return info
