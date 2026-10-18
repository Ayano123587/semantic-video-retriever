from unittest.mock import Mock
import json
import pytest
from src.models import evaluate


def test_demo_report_is_preserved_before_index_or_encoder_loading(tmp_path,monkeypatch):
    report=tmp_path/'existing.json';report.write_text('preserve recorded result')
    encoder=Mock(side_effect=AssertionError('no model allocation'))
    monkeypatch.setattr(evaluate,'CLIPEncoder',encoder)
    with pytest.raises(ValueError,match='Fresh'):evaluate.evaluate('absent index','absent annotations',report)
    assert report.read_text()=='preserve recorded result'
    encoder.assert_not_called()


def test_multi_video_demo_requires_target_identity_before_encoder(tmp_path,monkeypatch):
    index=Mock(records=[{'video_sha256':'a'},{'video_sha256':'b'}])
    monkeypatch.setattr(evaluate.VectorIndex,'load',lambda _:index)
    encoder=Mock(side_effect=AssertionError('no encoder before annotation validation'))
    monkeypatch.setattr(evaluate,'CLIPEncoder',encoder)
    annotations=tmp_path/'queries.json';annotations.write_text(json.dumps([{'query':'query','relevant_intervals_seconds':[[0,4]]}]))
    with pytest.raises(ValueError,match='target video'):evaluate.evaluate('index',annotations,tmp_path/'report.json')
    encoder.assert_not_called()


def test_matching_time_in_wrong_video_is_not_a_retrieval_hit(tmp_path,monkeypatch):
    index=Mock(records=[{'video_sha256':'a'},{'video_sha256':'b'}],metadata={})
    monkeypatch.setattr(evaluate.VectorIndex,'load',lambda _:index)
    monkeypatch.setattr(evaluate,'CLIPEncoder',lambda:object())
    monkeypatch.setattr(evaluate,'search',lambda *_:[{'video_sha256':'b','timestamp_seconds':2}])
    annotations=tmp_path/'queries.json';annotations.write_text(json.dumps([{'query':'query','video_sha256':'a','relevant_intervals_seconds':[[0,4]]}]))
    report=tmp_path/'report.json';evaluate.evaluate('index',annotations,report)
    result=json.loads(report.read_text())
    assert result['hit_at_1']==result['hit_at_3']==0
