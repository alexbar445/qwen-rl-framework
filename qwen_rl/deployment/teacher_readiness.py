"""Non-billable Teacher provider readiness shared by startup and resume checks."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Mapping


def teacher_provider_status(runtime: Mapping[str, Any], credential_files: Mapping) -> dict:
    bepa = runtime.get('bepa') or {}
    if not bepa.get('enabled'):
        return {'required': False, 'status': 'not_required'}
    from ..bepa.teacher import TeacherProviderConfig, TeacherRunner, TeacherRunnerError
    from .instance import _secret

    try:
        teacher = bepa['teacher']
        name = teacher.get('api_key_env', 'DEEPSEEK_API_KEY')
        key = os.environ.get(name)
        if not key:
            path = credential_files.get(name)
            if path is None:
                return {'required': True, 'status': 'blocked', 'reason_code': 'provider_credential_missing'}
            key = _secret(Path(path))
        runner = TeacherRunner(
            central_base_url='http://127.0.0.1',
            provider=TeacherProviderConfig(base_url=teacher['base_url'], model=teacher['model'],
                                           api_key_env=name, timeout_seconds=3.0),
        )
        asyncio.run(runner._probe_provider(key))
    except TeacherRunnerError as exc:
        return {'required': True, 'status': 'blocked', 'reason_code': exc.code}
    except Exception:
        # Provider exceptions/configuration can contain credentials and URLs.
        return {'required': True, 'status': 'blocked', 'reason_code': 'provider_readiness_unavailable'}
    return {'required': True, 'status': 'passed', 'scope': 'authenticated_model_catalog_no_completion'}
