from __future__ import annotations

import numpy as np

from music_insight import chords, midi_render, synth


def test_chords_on_synth_loop(tmp_path):
    wav = tmp_path / "loop.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)

    from music_insight import audio
    samples, sr = audio.load_audio(wav)
    result = chords.detect(samples, sr)

    assert len(result["frame_labels"]) > 10
    assert result["segments"]
    assert all("chord" in s for s in result["segments"])


def test_beat_sync_detection(tmp_path):
    wav = tmp_path / "loop.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)

    from music_insight import audio
    samples, sr = audio.load_audio(wav)
    beat_times = np.arange(0.0, len(samples) / sr, 0.6)  # 100 BPM
    result = chords.detect_beat_sync(samples, sr, beat_times)

    assert len(result["beat_labels"]) == len(beat_times) - 1
    assert result["segments"]
    assert result["beat_centers"][0] > 0


def test_midi_render_roundtrip(tmp_path):
    """渲染一段极简 MIDI（单音 C），确认输出有能量且时长合理。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)
    midi_path = tmp_path / "c.mid"
    pm.write(str(midi_path))

    audio, duration = midi_render.render_midi(str(midi_path))
    assert duration > 0.5
    assert audio.size > 1000
    assert float(abs(audio).max()) > 0.1
