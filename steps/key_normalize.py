import pretty_midi
from music21 import stream, note, pitch

MAJOR_LIKE = {"major", "ionian", "maj"}
MINOR_LIKE = {"minor", "aeolian", "min"}


def _collect_pitches(instruments):
    """
    Collect MIDI pitches from a dict[name -> pretty_midi.Instrument],
    skipping drums and obviously invalid notes.
    """
    pitches = []
    for name, inst in (instruments or {}).items():
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            if 0 < n.pitch < 128:
                pitches.append(int(n.pitch))
    return pitches


def _detect_key_music21(pitches):
    """
    Use music21's key analyzer on a synthetic stream built from MIDI pitches.
    Returns (tonic_str, mode_str) or (None, None).
    """
    if not pitches:
        return None, None

    s = stream.Stream()
    # Use dummy quarter notes at pitch classes; we only care about distribution.
    for p in pitches:
        try:
            s.append(note.Note(p % 128, quarterLength=1.0))
        except Exception:
            continue

    if len(s.notes) < 4:
        return None, None

    try:
        k = s.analyze("KrumhanslSchmuckler")
    except Exception:
        return None, None

    tonic = k.tonic.name if hasattr(k, "tonic") and k.tonic else None
    mode = k.mode.lower() if hasattr(k, "mode") and k.mode else None
    return tonic, mode


def _compute_transpose_semitones(tonic, mode):
    """
    Decide semitone transpose so that:
      - major-ish key -> C major
      - minor-ish key -> A minor

    Returns (semitones, target_key_str, reason) where:
      - semitones is a signed int in [-6, +6]
      - target_key_str is "C major" or "A minor" (or None if unknown)
      - reason is an optional string for manifest clarity
    """
    if not tonic or not mode:
        return 0, None, "missing tonic/mode"

    mode_norm = (mode or "").lower()
    if mode_norm in MAJOR_LIKE:
        target_tonic = "C"
        target_label = "C major"
    elif mode_norm in MINOR_LIKE:
        target_tonic = "A"
        target_label = "A minor"
    else:
        return 0, None, f"unsupported mode: {mode}"

    try:
        src_pc = pitch.Pitch(tonic).pitchClass
        tgt_pc = pitch.Pitch(target_tonic).pitchClass
    except Exception as e:
        return 0, None, f"pitch parse failed: {e}"

    # unsigned shift in [0, 11]
    shift = (tgt_pc - src_pc) % 12
    # choose a signed minimal-ish shift in [-6, +6]
    if shift > 6:
        shift -= 12

    if shift == 0:
        return 0, target_label, "already in target key"

    return int(shift), target_label, None


def _transpose_instruments(instruments, semitones):
    """
    Return a new dict[name -> pretty_midi.Instrument] with pitches shifted.
    Drums are not touched.
    """
    if not instruments or semitones == 0:
        return instruments

    out = {}
    for name, inst in instruments.items():
        if getattr(inst, "is_drum", False):
            out[name] = inst
            continue

        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name or name,
        )

        for n in inst.notes:
            new_pitch = int(n.pitch) + int(semitones)
            if 0 < new_pitch < 128:
                new_inst.notes.append(
                    pretty_midi.Note(
                        start=float(n.start),
                        end=float(n.end),
                        pitch=int(new_pitch),
                        velocity=int(n.velocity),
                    )
                )

        # Preserve CC / pitch bends (shallow copy is fine)
        for cc in getattr(inst, "control_changes", []):
            new_inst.control_changes.append(cc)
        for pb in getattr(inst, "pitch_bends", []):
            new_inst.pitch_bends.append(pb)

        out[name] = new_inst

    return out


def detect_key_only(assigned_instruments, manifest):
    """
    Detect key and write it into manifest["key"], without transposing.
    """
    pitches = _collect_pitches(assigned_instruments)
    tonic, mode = _detect_key_music21(pitches)

    key_info = manifest.setdefault("key", {})
    key_info["detected_tonic"] = tonic
    key_info["detected_mode"] = mode

    key_info["normalized"] = False
    key_info["transpose_semitones"] = 0
    key_info["target"] = None
    key_info["reason"] = "key detection only"

    return tonic, mode


def detect_and_normalize_key(assigned_instruments, CFG, manifest):
    """
    1. Detect global key from assigned instruments (ignoring drums).
    2. Transpose pitched tracks so that:
         - major-ish -> C major
         - minor-ish -> A minor
    3. Update manifest['key'] with detection + transpose info.
    4. Return (possibly) transposed instruments dict.
    """
    pitches = _collect_pitches(assigned_instruments)
    tonic, mode = _detect_key_music21(pitches)

    key_info = manifest.setdefault("key", {})
    key_info["detected_tonic"] = tonic
    key_info["detected_mode"] = mode

    semitones, target, reason = _compute_transpose_semitones(tonic, mode)

    # If we couldn't determine a target, do nothing
    if target is None:
        key_info["normalized"] = False
        key_info["transpose_semitones"] = 0
        key_info["target"] = None
        key_info["reason"] = reason or "could not determine target key"
        return assigned_instruments

    # If already in target, record that normalization is effectively satisfied
    if semitones == 0:
        key_info["normalized"] = True
        key_info["transpose_semitones"] = 0
        key_info["target"] = target
        key_info["reason"] = reason or "already in target key"
        return assigned_instruments

    normalized = _transpose_instruments(assigned_instruments, semitones)

    key_info["normalized"] = True
    key_info["transpose_semitones"] = int(semitones)
    key_info["target"] = target
    if reason:
        key_info["reason"] = reason
    else:
        key_info.pop("reason", None)

    return normalized
