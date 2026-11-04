"""
Text-to-Viseme Lip-Sync & Conversational Hand-Gesture Engine for Girl12 (CC3+ Rig)
Run inside Blender (Scripting tab -> Run Script) or import from an external voice/TTS pipeline.

Whenever she speaks (either from text or from voice phoneme timestamps):
- Her legs, hips, and feet stay completely still in a natural standing pose.
- Her mouth, lips, and jaw synchronize with the words.
- Her hands and forearms perform natural conversational gestures only while she is speaking,
  and rest calmly at her sides when she finishes speaking.
"""

from __future__ import annotations

import bpy
import math
import random
from mathutils import Euler, Quaternion

VIS_KEYS = [
    "V_Open",
    "V_Explosive",
    "V_Dental_Lip",
    "V_Tight_O",
    "V_Tight",
    "V_Wide",
    "V_Affricate",
    "V_Lip_Open",
]

CHAR_TO_VISEME = {
    "a": {"V_Open": 0.75, "V_Wide": 0.25},
    "e": {"V_Wide": 0.65, "V_Open": 0.30},
    "i": {"V_Wide": 0.80, "V_Open": 0.15},
    "o": {"V_Tight_O": 0.80, "V_Open": 0.35},
    "u": {"V_Tight_O": 0.85, "V_Tight": 0.30},
    "y": {"V_Wide": 0.60, "V_Open": 0.20},
    "b": {"V_Explosive": 0.90},
    "p": {"V_Explosive": 0.95},
    "m": {"V_Explosive": 0.85},
    "f": {"V_Dental_Lip": 0.85},
    "v": {"V_Dental_Lip": 0.80},
    "t": {"V_Dental_Lip": 0.35, "V_Wide": 0.30, "V_Lip_Open": 0.25},
    "d": {"V_Dental_Lip": 0.35, "V_Open": 0.25},
    "n": {"V_Dental_Lip": 0.30, "V_Lip_Open": 0.30},
    "l": {"V_Lip_Open": 0.55, "V_Open": 0.25},
    "s": {"V_Tight": 0.45, "V_Wide": 0.35},
    "z": {"V_Tight": 0.45, "V_Wide": 0.35},
    "c": {"V_Affricate": 0.65, "V_Wide": 0.20},
    "j": {"V_Affricate": 0.75},
    "g": {"V_Open": 0.35, "V_Lip_Open": 0.40},
    "k": {"V_Open": 0.35, "V_Lip_Open": 0.40},
    "q": {"V_Tight_O": 0.70, "V_Open": 0.20},
    "r": {"V_Tight": 0.55, "V_Lip_Open": 0.30},
    "w": {"V_Tight_O": 0.80},
    "h": {"V_Open": 0.35, "V_Lip_Open": 0.30},
}

DIGRAPH_TO_VISEME = {
    "th": {"V_Dental_Lip": 0.65, "V_Lip_Open": 0.35},
    "sh": {"V_Affricate": 0.85, "V_Lip_Open": 0.25},
    "ch": {"V_Affricate": 0.85, "V_Lip_Open": 0.25},
    "oo": {"V_Tight_O": 0.90, "V_Open": 0.25},
    "ee": {"V_Wide": 0.85, "V_Open": 0.25},
    "ea": {"V_Wide": 0.85, "V_Open": 0.25},
}

FINGER_BASE = {
    "CC_Base_L_Pinky1": (0.983, -0.041, -0.009, -0.179),
    "CC_Base_L_Pinky2": (0.939, -0.001, 0.009, -0.345),
    "CC_Base_L_Pinky3": (0.992, 0.021, 0.007, -0.122),
    "CC_Base_L_Ring1": (0.983, -0.042, -0.008, -0.178),
    "CC_Base_L_Ring2": (0.941, -0.001, 0.014, -0.338),
    "CC_Base_L_Ring3": (0.993, 0.009, 0.003, -0.12),
    "CC_Base_L_Mid1": (0.983, -0.043, -0.01, -0.178),
    "CC_Base_L_Mid2": (0.969, 0.002, 0.003, -0.248),
    "CC_Base_L_Mid3": (0.995, 0.01, 0.001, -0.095),
    "CC_Base_L_Index1": (0.996, -0.043, -0.005, -0.078),
    "CC_Base_L_Index2": (0.99, 0.001, -0.003, -0.143),
    "CC_Base_L_Index3": (0.998, 0.003, -0.0, -0.066),
    "CC_Base_L_Thumb1": (0.983, -0.141, -0.023, -0.119),
    "CC_Base_L_Thumb2": (0.999, -0.042, -0.011, 0.002),
    "CC_Base_L_Thumb3": (1.0, -0.029, -0.001, -0.01),
    "CC_Base_R_Ring1": (0.974, -0.041, 0.009, 0.223),
    "CC_Base_R_Ring2": (0.898, -0.001, -0.018, 0.439),
    "CC_Base_R_Ring3": (0.974, 0.009, -0.008, 0.228),
    "CC_Base_R_Mid1": (0.974, -0.042, 0.01, 0.223),
    "CC_Base_R_Mid2": (0.936, 0.002, -0.006, 0.352),
    "CC_Base_R_Mid3": (0.979, 0.01, -0.003, 0.203),
    "CC_Base_R_Thumb1": (0.983, -0.141, 0.023, 0.119),
    "CC_Base_R_Thumb2": (0.999, -0.042, 0.011, -0.002),
    "CC_Base_R_Thumb3": (1.0, -0.029, 0.001, 0.01),
    "CC_Base_R_Index1": (0.991, -0.043, 0.005, 0.123),
    "CC_Base_R_Index2": (0.968, 0.001, 0.003, 0.25),
    "CC_Base_R_Index3": (0.985, 0.003, -0.0, 0.174),
    "CC_Base_R_Pinky1": (0.974, -0.041, 0.009, 0.223),
    "CC_Base_R_Pinky2": (0.895, -0.001, -0.011, 0.445),
    "CC_Base_R_Pinky3": (0.973, 0.021, -0.015, 0.229),
}


def set_viseme_frame(kbs, frame: int, weights: dict):
    """Set all 8 CC3+ viseme shape keys at the given frame."""
    for vk in VIS_KEYS:
        kbs[vk].value = weights.get(vk, 0.0)
        kbs[vk].keyframe_insert(data_path="value", frame=frame)


def generate_standing_hand_gestures(
    start_speech_frame: int,
    end_speech_frame: int,
    action_name: str = "Body_Stand_And_Talk",
):
    """
    Generate a clean standing posture with natural conversational hand/forearm gestures
    synchronized to the speech window [start_speech_frame, end_speech_frame].
    Legs and hips remain completely stationary.
    """
    arm = bpy.data.objects.get("Armature")
    if not arm:
        return

    if action_name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[action_name])

    stand_act = bpy.data.actions.new(name=action_name)
    stand_act.use_fake_user = True
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = stand_act

    # Reset all bones to stationary standing rest pose
    for pb in arm.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)

    def key_bone_quat(bname, quat, frame):
        pb = arm.pose.bones[bname]
        pb.rotation_quaternion = quat
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    total_frames = end_speech_frame + 24

    for f in range(1, total_frames + 1, 8):
        if f < start_speech_frame:
            env = max(0.0, (f - 1) / max(1.0, float(start_speech_frame - 1)))
        elif f > end_speech_frame:
            env = max(0.0, 1.0 - (f - end_speech_frame) / 20.0)
        else:
            env = 1.0

        t = f / 24.0

        # Subtle upper-spine breathing and gentle head nod/turn while talking
        spine_pitch = 0.015 * math.sin(t * 1.8)
        head_nod = env * (0.035 * math.sin(t * 3.2) + 0.02 * math.cos(t * 1.9))
        head_yaw = env * (0.045 * math.sin(t * 1.1))
        head_tilt = env * (0.025 * math.sin(t * 1.5 + 0.8))

        key_bone_quat("CC_Base_Spine02", Euler((spine_pitch, 0.0, 0.0), "XYZ").to_quaternion(), f)
        key_bone_quat("CC_Base_Head", Euler((head_nod, head_tilt, head_yaw), "XYZ").to_quaternion(), f)

        # Relaxed shoulders
        key_bone_quat("CC_Base_L_Clavicle", Euler((0.0, 0.0, -0.22), "XYZ").to_quaternion(), f)
        key_bone_quat("CC_Base_R_Clavicle", Euler((0.0, 0.0, 0.22), "XYZ").to_quaternion(), f)

        # Conversational hand waves (active only while speaking)
        l_wave = max(0.0, math.sin(t * 1.7) + 0.4 * math.sin(t * 3.1)) * 0.5
        r_wave = max(0.0, math.sin(t * 1.4 + 1.8) + 0.4 * math.cos(t * 2.7)) * 0.5

        l_up_x = env * (0.10 + 0.08 * l_wave)
        l_up_z = -0.45 + env * (0.04 * l_wave)
        key_bone_quat("CC_Base_L_Upperarm", Euler((l_up_x, -0.10 * env, l_up_z), "XYZ").to_quaternion(), f)

        r_up_x = env * (0.10 + 0.08 * r_wave)
        r_up_z = 0.45 - env * (0.04 * r_wave)
        key_bone_quat("CC_Base_R_Upperarm", Euler((r_up_x, 0.10 * env, r_up_z), "XYZ").to_quaternion(), f)

        l_fore_x = 0.25 + env * (0.65 + 0.30 * math.sin(t * 2.6))
        l_fore_y = env * (0.22 + 0.15 * math.sin(t * 2.1))
        l_fore_z = -0.10 - env * (0.15 + 0.12 * math.cos(t * 2.6))
        key_bone_quat("CC_Base_L_Forearm", Euler((l_fore_x, l_fore_y, l_fore_z), "XYZ").to_quaternion(), f)

        r_fore_x = 0.25 + env * (0.60 + 0.30 * math.sin(t * 2.3 + 1.2))
        r_fore_y = -env * (0.22 + 0.15 * math.cos(t * 1.9))
        r_fore_z = 0.10 + env * (0.15 + 0.12 * math.sin(t * 2.3 + 1.2))
        key_bone_quat("CC_Base_R_Forearm", Euler((r_fore_x, r_fore_y, r_fore_z), "XYZ").to_quaternion(), f)

        l_hand_x = env * (0.12 * math.sin(t * 3.8))
        l_hand_y = env * (-0.18 + 0.15 * math.cos(t * 3.1))
        l_hand_z = env * (0.12 * math.sin(t * 2.6))
        key_bone_quat("CC_Base_L_Hand", Euler((l_hand_x, l_hand_y, l_hand_z), "XYZ").to_quaternion(), f)

        r_hand_x = env * (0.12 * math.cos(t * 3.5))
        r_hand_y = env * (0.18 + 0.15 * math.sin(t * 2.9))
        r_hand_z = env * (-0.12 * math.cos(t * 2.3))
        key_bone_quat("CC_Base_R_Hand", Euler((r_hand_x, r_hand_y, r_hand_z), "XYZ").to_quaternion(), f)

        finger_open_l = env * 0.08 * math.sin(t * 3.2)
        finger_open_r = env * 0.08 * math.cos(t * 2.9)
        for fname, base_q in FINGER_BASE.items():
            q = Quaternion(base_q)
            if "Thumb" not in fname:
                delta = finger_open_l if "_L_" in fname else -finger_open_r
                q = q @ Euler((0.0, 0.0, delta), "XYZ").to_quaternion()
            key_bone_quat(fname, q, f)


def animate_from_text(
    text: str,
    start_frame: int = 12,
    frames_per_char: float = 2.2,
    action_name: str = "Face_Speech_And_Expressions",
    add_blinks_and_expressions: bool = True,
    sync_hand_gestures: bool = True,
) -> int:
    """
    Generate keyframed lip-sync, facial expressions, and conversational hand gestures from text.
    """
    body = bpy.data.objects.get("CC_Base_Body")
    if not body or not body.data.shape_keys:
        raise RuntimeError("CC_Base_Body with shape keys not found in scene.")

    sk_data = body.data.shape_keys
    kbs = sk_data.key_blocks

    if not sk_data.animation_data:
        sk_data.animation_data_create()

    if action_name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[action_name])

    face_act = bpy.data.actions.new(name=action_name)
    face_act.use_fake_user = True
    sk_data.animation_data.action = face_act

    for kb in kbs[1:]:
        kb.value = 0.0
        kb.keyframe_insert(data_path="value", frame=1)

    cur_frame = float(start_frame)
    set_viseme_frame(kbs, max(1, int(cur_frame - 2)), {})

    i = 0
    text_lower = text.lower()
    while i < len(text_lower):
        ch = text_lower[i]
        digraph = text_lower[i : i + 2]

        if digraph in DIGRAPH_TO_VISEME:
            weights = DIGRAPH_TO_VISEME[digraph]
            step = 2
            dur = frames_per_char * 1.5
        elif ch in CHAR_TO_VISEME:
            weights = CHAR_TO_VISEME[ch]
            step = 1
            dur = frames_per_char * (1.3 if ch in "aeiou" else 0.9)
        elif ch in (" ", ",", ".", "!", "?", ";", ":"):
            weights = {}
            step = 1
            dur = frames_per_char * (1.5 if ch == " " else 3.5)
        else:
            i += 1
            continue

        f_int = int(round(cur_frame))
        set_viseme_frame(kbs, f_int, weights)
        cur_frame += dur
        i += step

    end_speech_frame = int(round(cur_frame + 2))
    set_viseme_frame(kbs, end_speech_frame, {})

    if add_blinks_and_expressions:
        total_frames = max(end_speech_frame + 24, 250)
        random.seed(42)
        f = 35
        while f < total_frames:
            for bk in ["Eye_Blink_L", "Eye_Blink_R"]:
                kbs[bk].value = 0.0
                kbs[bk].keyframe_insert(data_path="value", frame=f - 3)
                kbs[bk].value = 1.0
                kbs[bk].keyframe_insert(data_path="value", frame=f)
                kbs[bk].value = 0.0
                kbs[bk].keyframe_insert(data_path="value", frame=f + 4)
            f += random.randint(65, 115)

        for f in range(1, total_frames, 25):
            smile_val = 0.22 + 0.12 * math.sin(f * 0.04)
            kbs["Mouth_Smile_L"].value = smile_val
            kbs["Mouth_Smile_R"].value = smile_val
            kbs["Mouth_Smile_L"].keyframe_insert(data_path="value", frame=f)
            kbs["Mouth_Smile_R"].keyframe_insert(data_path="value", frame=f)

            cheek_val = smile_val * 0.5
            kbs["Cheek_Raise_L"].value = cheek_val
            kbs["Cheek_Raise_R"].value = cheek_val
            kbs["Cheek_Raise_L"].keyframe_insert(data_path="value", frame=f)
            kbs["Cheek_Raise_R"].keyframe_insert(data_path="value", frame=f)

            brow_val = max(0.0, 0.35 * math.sin(f * 0.07)) if f < end_speech_frame else 0.0
            for br in [
                "Brow_Raise_Inner_L",
                "Brow_Raise_Inner_R",
                "Brow_Raise_Outer_L",
                "Brow_Raise_Outer_R",
            ]:
                kbs[br].value = brow_val
                kbs[br].keyframe_insert(data_path="value", frame=f)

    if sync_hand_gestures:
        generate_standing_hand_gestures(start_frame, end_speech_frame)

    bpy.context.scene.frame_end = end_speech_frame + 24
    return end_speech_frame


def animate_from_phoneme_timestamps(
    events: list[dict],
    fps: int = 24,
    action_name: str = "Face_Voice_Synced",
    sync_hand_gestures: bool = True,
) -> int:
    """
    Hook for the Voice Model / TTS developer:
    Pass a list of timestamped phoneme/viseme events:
        [{"start": 0.0, "end": 0.12, "token": "h"}, {"start": 0.12, "end": 0.28, "token": "e"}, ...]
    or direct viseme dicts:
        [{"start": 0.0, "end": 0.15, "visemes": {"V_Open": 0.8, "V_Wide": 0.2}}]
    """
    body = bpy.data.objects.get("CC_Base_Body")
    sk_data = body.data.shape_keys
    kbs = sk_data.key_blocks

    if not sk_data.animation_data:
        sk_data.animation_data_create()

    if action_name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[action_name])

    face_act = bpy.data.actions.new(name=action_name)
    face_act.use_fake_user = True
    sk_data.animation_data.action = face_act

    set_viseme_frame(kbs, 1, {})
    first_frame = None
    last_frame = 1

    for ev in events:
        f_start = max(1, int(round(ev["start"] * fps)))
        f_end = max(f_start + 1, int(round(ev["end"] * fps)))
        f_mid = (f_start + f_end) // 2
        if first_frame is None:
            first_frame = f_start

        if "visemes" in ev:
            weights = ev["visemes"]
        else:
            tok = ev.get("token", "").lower()
            weights = DIGRAPH_TO_VISEME.get(tok) or CHAR_TO_VISEME.get(tok, {})

        set_viseme_frame(kbs, f_mid, weights)
        last_frame = max(last_frame, f_end)

    end_speech_frame = last_frame + 2
    set_viseme_frame(kbs, end_speech_frame, {})

    if sync_hand_gestures:
        generate_standing_hand_gestures(first_frame or 1, end_speech_frame)

    bpy.context.scene.frame_end = end_speech_frame + 24
    return end_speech_frame


def load_audio_and_sync(
    audio_path: str,
    events: list[dict] | None = None,
    transcript_text: str | None = None,
    start_frame: int = 12,
    fps: int = 24,
) -> int:
    """
    Complete Voice Model + Blender integration helper:
    1. Attaches the generated voice audio (.wav / .mp3) to Blender's Video Sequence Editor
       at `start_frame` and enables Audio Scrubbing & AV-Sync so playback has live sound.
    2. Drives the character's mouth, jaw, facial expressions, and hand gestures using either:
       - `events`: timestamped phoneme/viseme list from the voice model / forced aligner
       - `transcript_text`: fallback text lip-sync stretched to match the audio duration.
    """
    import os
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.use_audio_scrub = True
    scene.sync_mode = 'AUDIO_SYNC'

    if not scene.sequence_editor:
        scene.sequence_editor_create()

    # Remove previous voice strips
    seq = scene.sequence_editor
    strips = getattr(seq, "strips", None) or getattr(seq, "sequences", None)
    for s in list(strips):
        if s.name.startswith("Voice_Audio"):
            strips.remove(s)

    audio_duration_frames = 0
    if audio_path and os.path.exists(audio_path):
        sound_strip = strips.new_sound(
            name="Voice_Audio",
            filepath=audio_path,
            channel=1,
            frame_start=start_frame,
        )
        audio_duration_frames = int(sound_strip.frame_final_duration)

    if events:
        # Offset events by start_frame / fps
        offset_sec = start_frame / float(fps)
        shifted = [{**ev, "start": ev["start"] + offset_sec, "end": ev["end"] + offset_sec} for ev in events]
        return animate_from_phoneme_timestamps(shifted, fps=fps, sync_hand_gestures=True)
    elif transcript_text:
        # Calculate frames_per_char so the text finishes right when the audio clip ends
        if audio_duration_frames > 0 and len(transcript_text) > 0:
            fpc = max(1.0, audio_duration_frames / (len(transcript_text) * 1.25))
        else:
            fpc = 2.2
        return animate_from_text(
            transcript_text,
            start_frame=start_frame,
            frames_per_char=fpc,
            sync_hand_gestures=True,
        )
    return start_frame + audio_duration_frames


if __name__ == "__main__":
    TEST_TEXT = (
        "Hello there! Welcome to our project. "
        "Right now you are watching a test of my procedural lip sync and facial animation system in Blender. "
        "As I read this text, my mouth, jaw, eyebrows, and hands move naturally with the words!"
    )
    end_f = animate_from_text(TEST_TEXT, start_frame=12, frames_per_char=2.2)
    print(f"Animated lip sync and hand gestures for text up to frame {end_f}")
