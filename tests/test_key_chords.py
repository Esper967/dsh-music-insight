from __future__ import annotations

from music_insight import keydetect


def test_relative_pairs():
    assert keydetect._relative_of("F 大调") == "D 小调"
    assert keydetect._relative_of("G 小调") == "A# 大调"
    assert keydetect._relative_of("C 大调") == "A 小调"


def test_key_from_chords_flags_relative_ambiguity():
    # F 大调和 D 小调共享和弦，必须并列报出来
    chords = [("F", 60.0), ("Dm", 40.0), ("C", 20.0), ("A#", 15.0)]
    result = keydetect.key_from_chords(chords)
    assert result["ambiguous_relative"] is True
    assert set(result["relative_pair"]) == {"F 大调", "D 小调"}


def test_key_from_chords_clear_winner():
    # 大量主和弦 + 主和弦持续最久：不该标歧义
    chords = [("C", 200.0), ("G", 20.0), ("F", 20.0), ("Am", 10.0)]
    result = keydetect.key_from_chords(chords)
    assert result["key"] == "C 大调"
