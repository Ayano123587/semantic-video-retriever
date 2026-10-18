"""Execution identity mechanics; no encoder allocation or benchmark data."""
import platform
from src.models import moment_encoder


def test_cpu_identity_records_first_processor_isa_without_volatile_fields(tmp_path,monkeypatch):
    source=tmp_path/'cpuinfo'
    source.write_text('processor: 0\nvendor_id: Vendor\nmodel name: Model\nflags: avx avx2\ncpu MHz: 1234\n\nprocessor: 1\nmodel name: Other\n',encoding='utf-8')
    monkeypatch.setattr(moment_encoder,'Path',lambda _:source)
    monkeypatch.setattr(platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(platform,'processor',lambda:'')
    value=moment_encoder.cpu_identity()
    assert value=={'machine':'x86_64','processor':'','cpuinfo':{'vendor_id':'Vendor','model name':'Model','flags':'avx avx2'}}
    source.write_text(source.read_text().replace('1234','4321'),encoding='utf-8')
    assert moment_encoder.cpu_identity()==value


def test_cpu_identity_without_procfs_preserves_platform_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(moment_encoder,'Path',lambda _:tmp_path/'missing')
    monkeypatch.setattr(platform,'machine',lambda:'AMD64')
    monkeypatch.setattr(platform,'processor',lambda:'CPU model')
    assert moment_encoder.cpu_identity()=={'machine':'AMD64','processor':'CPU model'}
