import copy
import pytest
from manager.spec import build_spec


def detection():
    return {'os': 'linux', 'arch': 'x86_64', 'backend': 'cuda', 'python': '3.11.8',
            'nvidia': {'driver': '580.105.08', 'cuda_version': '13.0'}}


def test_opt_in_profile(monkeypatch):
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    spec = build_spec(detection())
    assert spec.backend == 'cu134' and spec.python_version == '3.12' and not spec.spark_native
    assert set(spec.torch_constraints()) >= {'torch==2.15.0+cu134', 'torchvision==0.29.2+cu134', 'torchaudio==2.11.1+cu134'}
    assert {'triton==3.9.0', 'bitsandbytes==0.50.2', 'torchcodec==0.15.0'} <= set(spec.extra_packages)
    assert not any('flash_attn' in x.lower() or 'flash-attn' in x.lower() or 'natten' in x.lower() for x in spec.optional_packages)


@pytest.mark.parametrize('key,value', [('os', 'windows'), ('arch', 'aarch64'), ('backend', 'cpu')])
def test_incompatible_profile(monkeypatch, key, value):
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    data = detection(); data[key] = value
    with pytest.raises(RuntimeError): build_spec(data)


@pytest.mark.parametrize('key,value', [('driver', '579.1'), ('cuda_version', '12.9')])
def test_incompatible_driver(monkeypatch, key, value):
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    data = detection(); data['nvidia'][key] = value
    with pytest.raises(RuntimeError): build_spec(data)


def test_unknown_profile_rejected(monkeypatch):
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'typo')
    with pytest.raises(RuntimeError, match='Unknown'): build_spec(detection())


def test_profile_installed_package_health(monkeypatch):
    from manager import env
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    spec = build_spec(detection())
    assert set(spec.incompatible_packages) == {'flash-attn', 'natten'}
    assert set(spec.torch_constraints()) >= {'triton==3.9.0', 'bitsandbytes==0.50.2', 'torchcodec==0.15.0'}
    monkeypatch.setattr(env, '_installed_distributions', lambda packages: ['natten'])
    assert not env.incompatible_packages_absent(spec)
    monkeypatch.setattr(env, '_installed_distributions', lambda packages: [])
    assert env.incompatible_packages_absent(spec)
    old = detection(); old['python'] = '3.8.0'
    assert build_spec(old).python_version == '3.12'


def test_sync_rejects_old_managed_python(monkeypatch):
    from manager import env, uvbin, gitwin
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    spec = build_spec(detection())
    monkeypatch.setattr(uvbin, 'ensure_uv', lambda **kwargs: None)
    monkeypatch.setattr(gitwin, 'ensure_git', lambda **kwargs: None)
    monkeypatch.setattr(env, 'ensure_venv', lambda *args, **kwargs: None)
    monkeypatch.setattr(env, 'venv_exists', lambda: True)
    monkeypatch.setattr(env, 'venv_python_version', lambda: '3.10')
    with pytest.raises(RuntimeError, match='environment Python 3.11'):
        env.sync(spec, detection())


def test_stale_binary_cleanup_and_default_spark_contract(monkeypatch):
    from manager import env
    from manager.spec import _make_spark_spec
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    spec = build_spec(detection())
    states = iter([['flash-attn', 'natten'], []])
    monkeypatch.setattr(env, '_installed_distributions', lambda names: next(states))
    removed = []
    monkeypatch.setattr(env, '_pip_uninstall', lambda names, dry_run=False: removed.extend(names))
    env._remove_incompatible_packages(spec)
    assert removed == ['flash-attn', 'natten']
    monkeypatch.delenv('AI_TOOLKIT_TORCH_PROFILE')
    default = build_spec(detection())
    assert not default.required_packages and not default.incompatible_packages
    spark = _make_spark_spec({}, '/fixture/wheels', [])
    assert spark.spark_native and not spark.required_packages and not spark.incompatible_packages


def test_required_extra_version_drift_forces_reinstall(monkeypatch):
    import json
    from types import SimpleNamespace
    from manager import env
    monkeypatch.setenv('AI_TOOLKIT_TORCH_PROFILE', 'torch215')
    spec = build_spec(detection())
    monkeypatch.setattr(env, 'venv_exists', lambda: True)
    monkeypatch.setattr(env, 'venv_python', lambda: '/fake/python')
    monkeypatch.setattr(env, 'clean_env', lambda: {})
    versions = dict(spec.required_packages)
    def run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(versions).encode())
    monkeypatch.setattr(env.subprocess, 'run', run)
    assert env.required_packages_match(spec)
    versions['torchcodec'] = '0.9.1'
    assert not env.required_packages_match(spec)
    monkeypatch.setattr(env, 'requirements_in_sync', lambda spec: True)
    monkeypatch.setattr(env, '_stale_git_pins', lambda spec: [])
    monkeypatch.setattr(env, '_torch_pin_args', lambda spec, dry_run=False: [])
    monkeypatch.setattr(env, '_pip_install', lambda *args, **kwargs: 0)
    monkeypatch.setattr(env, '_filter_extras', lambda extras: [])
    monkeypatch.setattr(env, '_verify_torch', lambda *args, **kwargs: None)
    assert env.ensure_requirements(spec, dry_run=True)
