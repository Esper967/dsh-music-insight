from __future__ import annotations

import numpy as np

from music_insight import audio, beats, melody, midi_render, synth


def test_beat_tracking_on_synth_loop(tmp_path):
    wav = tmp_path / "loop.wav"
    duration = synth.synth_demo(str(wav), bpm=100.0, bars=8)
    samples, sr = audio.load_audio(wav)

    tracked = beats.track_beats(samples, sr)
    assert len(tracked) >= 8
    intervals = np.diff(tracked)
    assert 0.5 < float(np.median(intervals)) < 0.75  # 100 BPM ≈ 0.6s
    assert len(tracked) <= duration / 0.6 + 5


def test_melody_tracking_single_note(tmp_path):
    """一段单音旋律，跟踪结果应落在目标音高附近。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="MELODY")
    for i, pitch in enumerate([69, 71, 72, 74]):
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch,
                                           start=i * 0.5, end=i * 0.5 + 0.45))
    pm.instruments.append(inst)
    midi = tmp_path / "melody.mid"
    pm.write(str(midi))

    audio_data, _ = midi_render.render_midi(str(midi))
    result = melody.track_melody(audio_data, midi_render.SR_DEFAULT)
    voiced = result["f0_midi"] > 0
    assert voiced.mean() > 0.3
    assert abs(float(np.median(result["f0_midi"][voiced])) - 72.0) < 6.0
