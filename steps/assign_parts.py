# steps/assign_parts.py
import pretty_midi

CANONICAL = ["drums", "voxlead", "voxbg", "bass", "guitar", "keys", "other"]

ALIASES = {
    # vocals
    "leadvox": "voxlead",
    "vox_lead": "voxlead",
    "vocal_lead": "voxlead",
    "voxlead": "voxlead",

    "voxbg": "voxbg",
    "bgvox": "voxbg",
    "backgroundvox": "voxbg",
    "vox_harm": "voxbg",
    "voxharm": "voxbg",
    "harmonyvox": "voxbg",
    "auxvox": "voxbg",   # NOTE: mapped into voxbg (no separate aux track today)

    # instruments
    "bass": "bass",
    "guitar": "guitar",
    "keys": "keys",
    "piano": "keys",

    "other": "other",
    "synth": "other",
    "pad": "other",
}

DEFAULT_TRACKS = ["drums", "voxlead", "voxbg", "bass", "guitar", "other"]  # keys optional

def normalize_tracks(tracks):
    """
    Normalize requested track names:
    - applies aliases
    - drops unknowns
    - de-dupes preserving order
    """
    if not tracks:
        return list(DEFAULT_TRACKS)

    out = []
    seen = set()
    for t in tracks:
        if not isinstance(t, str):
            continue
        k = t.strip().lower()
        if not k:
            continue
        k = ALIASES.get(k, k)
        if k not in CANONICAL:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out or list(DEFAULT_TRACKS)

def assign_tracks(pitched_midis, drums_inst, stems, CFG, manifest, tracks=None):
    """
    Filter/routable assignment into requested canonical tracks.
    Returns: dict[name -> pretty_midi.Instrument] used by assemble_and_write_midi.
    """
    cfg_tracks = None
    if isinstance(CFG, dict):
        # prefer CFG["tracks"], but keep compatibility with older CFG["classes"]
        cfg_tracks = CFG.get("tracks") or CFG.get("classes")

    if tracks is None:
        tracks = cfg_tracks

    # allow comma-separated string
    if isinstance(tracks, str):
        tracks = [x for x in tracks.split(",") if x.strip()]

    tracks_norm = normalize_tracks(tracks)

    assigned = {}

    # --- DRUMS ---
    if "drums" in tracks_norm and drums_inst is not None:
        drums_inst.is_drum = True
        drums_inst.name = "drums"
        assigned["drums"] = drums_inst

    # --- PITCHED CLASSES ---
    for name in [t for t in tracks_norm if t != "drums"]:
        inst = (pitched_midis or {}).get(name)
        if inst is None:
            continue
        notes = getattr(inst, "notes", [])
        if not notes:
            continue
        inst.name = name
        assigned[name] = inst

    manifest.setdefault("assignment", {})
    manifest["assignment"]["requested_tracks"] = tracks_norm
    manifest["assignment"]["tracks"] = list(assigned.keys())
    return assigned

# Backward-compatible name used by pipeline.py today
def assign_seven_classes(pitched_midis, drums_inst, stems, CFG, manifest):
    return assign_tracks(pitched_midis, drums_inst, stems, CFG, manifest, tracks=None)
