"""Authenticated control endpoints; teacher routes never expose the private scorer."""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import hmac
import os
from typing import Any, Literal
from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .contracts import BepaError, LeaseLostError, ConflictError

class RequestModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

class Claim(RequestModel):
    worker_id: str = Field(min_length=1, max_length=100)
    claim_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")

class Fence(RequestModel):
    fence: int = Field(ge=1)

class Heartbeat(Fence):
    worker_id: str = Field(min_length=1, max_length=100)

class Tool(Fence):
    sequence: int = Field(ge=0)
    name: str
    arguments: dict[str, Any]
    generated_text: str = ''
    completion_tokens: int = Field(ge=0)
    usage: dict[str, Any] | None = None
    call_id: str | None = None
    session_id: str | None = None

class Finish(Fence):
    session_id: str | None = None
    termination: str | None = None
    usage_calls: list[dict] | None = None
    token_total: int | None = None
    generation_stop_reason: Literal['eos', 'stop'] | None = None
    completion_tokens: int | None = Field(default=None, ge=0)
    usage: dict[str, Any] | None = None
    generated_text: str | None = None

    @model_validator(mode='after')
    def model_stop_has_usage(self):
        if self.termination == 'model_stop' and (
                self.generation_stop_reason is None or self.completion_tokens is None
                or self.usage is None):
            raise ValueError('model_stop requires the final generation stop and provider usage')
        return self

class Complete(Heartbeat):
    session_id: str
    native_messages: list[dict]
    plan: str | None
    plan_response: dict | None
    usage_calls: list[dict]
    token_total: int | None
    usage_known: bool
    termination: str
    source_receipt: dict | None
    source_accepted: bool
    provider: str
    requested_model: str
    actual_models: list[str]

class Fail(Heartbeat):
    reason: str | None = None
    termination: str | None = None
    token_total: int | None = None
    usage_tokens: int | None = None
    usage_calls: list[dict] = Field(default_factory=list)
    usage_known: bool = True
    outcome_unknown: bool = False
    error: str | None = None
    evidence: dict | None = None

class ProviderReceipt(Heartbeat):
    receipt: dict

class Bootstrap(RequestModel):
    task_ids: list[str] = Field(default_factory=list)

class BudgetUpdate(RequestModel):
    max_job_tokens: int | None = Field(default=None, ge=1, le=10_000_000)
    max_total_tokens: int | None = Field(default=None, ge=1, le=100_000_000)

class UsageReconciliation(RequestModel):
    evidence_ref: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

class LowerBoundSettlement(UsageReconciliation):
    accept_unreported_usage: Literal[True]
    reason: str = Field(min_length=1, max_length=500)


def attach_routes(app, coordinator, management_auth, bootstrap, *, status_context):
    def controller():
        if coordinator is None:
            raise HTTPException(409, 'BEPA is disabled in this runtime configuration')
        return coordinator

    def teacher_auth(authorization: str | None = Header(default=None)):
        ctrl = controller()
        token = os.environ.get(ctrl.settings['teacher']['token_env'])
        supplied = authorization.removeprefix('Bearer ') if authorization else ''
        if not token or not hmac.compare_digest(supplied, token):
            raise HTTPException(401, 'Teacher role token required')

    async def invoke(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except LeaseLostError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (BepaError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/bepa/status', dependencies=[Depends(management_auth)])
    async def status():
        if coordinator is None:
            return {'enabled': False}
        context, observed = status_context()
        result = await invoke(coordinator.status)
        result['diagnostics'] = await invoke(coordinator.diagnostics, **context)
        result['diagnostics']['current_run'].update(observed)
        return result

    @app.post('/bepa/budget', dependencies=[Depends(management_auth)])
    async def update_budget(request: BudgetUpdate):
        if request.max_job_tokens is None and request.max_total_tokens is None:
            raise HTTPException(422, 'at least one budget limit is required')
        return await invoke(coordinator.update_budget_limits,
                            max_job_tokens=request.max_job_tokens,
                            max_total_tokens=request.max_total_tokens)

    @app.get('/bepa/jobs', dependencies=[Depends(management_auth)])
    async def jobs():
        return [asdict(row) for row in await invoke(controller().store.list_jobs, newest_first=True)]

    @app.post('/bepa/jobs/{job_id}/reconcile-usage', dependencies=[Depends(management_auth)])
    async def reconcile_usage(job_id: str, request: UsageReconciliation):
        return await invoke(controller().reconcile_unknown_usage, job_id, request.evidence_ref)

    @app.post('/bepa/jobs/{job_id}/settle-lower-bound', dependencies=[Depends(management_auth)])
    async def settle_lower_bound(job_id: str, request: LowerBoundSettlement):
        return await invoke(controller().settle_unknown_lower_bound, job_id,
                            request.evidence_ref, request.reason)

    @app.get('/bepa/plans', dependencies=[Depends(management_auth)])
    async def plans():
        ctrl = controller()
        return await invoke(ctrl.store.list_retained_plans, list(ctrl.task_keys.values()),
                            engine_epoch=ctrl.epoch)

    @app.post('/bepa/plans/{plan_id}/forget', dependencies=[Depends(management_auth)])
    async def forget_plan(plan_id: str):
        return await invoke(controller().store.forget_plan, plan_id)

    @app.post('/bepa/bootstrap', status_code=202, dependencies=[Depends(management_auth)])
    async def start(request: Bootstrap):
        controller()
        return await bootstrap(request.task_ids)

    @app.post('/bepa/jobs/claim', dependencies=[Depends(teacher_auth)])
    async def claim(request: Claim):
        return {'job': await invoke(controller().claim_job, request.worker_id,
                                     claim_id=request.claim_id)}

    @app.post('/bepa/jobs/{job_id}/provider-receipt', dependencies=[Depends(teacher_auth)])
    async def provider_receipt(job_id: str, request: ProviderReceipt):
        return await invoke(controller().record_provider_receipt, job_id, request.model_dump())

    @app.post('/bepa/jobs/{job_id}/heartbeat', dependencies=[Depends(teacher_auth)])
    async def heartbeat(job_id: str, request: Heartbeat):
        ctrl = controller()
        result = await invoke(ctrl.store.heartbeat_job, job_id, worker_id=request.worker_id,
            fence=request.fence, lease_seconds=ctrl.settings['teacher']['budget']['job_timeout_seconds'])
        return {'job_id': job_id, 'status': 'leased'}

    @app.post('/bepa/jobs/{job_id}/source-start', dependencies=[Depends(teacher_auth)])
    async def source_start(job_id: str, request: Fence):
        return await invoke(controller().source_start, job_id, request.model_dump())

    @app.post('/bepa/jobs/{job_id}/tool', dependencies=[Depends(teacher_auth)])
    async def tool(job_id: str, request: Tool):
        try:
            return await controller().source_tool(job_id, request.model_dump(exclude_none=True))
        except (BepaError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/bepa/jobs/{job_id}/source-finish', dependencies=[Depends(teacher_auth)])
    async def finish(job_id: str, request: Finish):
        try:
            return await controller().source_finish(job_id, request.model_dump(exclude_none=True))
        except LeaseLostError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (BepaError, ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/bepa/jobs/{job_id}/complete', dependencies=[Depends(teacher_auth)])
    async def complete(job_id: str, request: Complete):
        return await invoke(controller().complete_job, job_id, request.model_dump())

    @app.post('/bepa/jobs/{job_id}/fail', dependencies=[Depends(teacher_auth)])
    async def fail(job_id: str, request: Fail):
        ctrl = controller()
        # An unknown last call must not erase the confirmed earlier call receipts.
        evidence_ref = await invoke(ctrl.artifacts.write_json,
            {'schema': 'bepa-failed-teacher-v1', 'job_id': job_id, 'payload': request.model_dump()})
        retry_source = (request.reason == 'source_rejected' and request.usage_known
            and not request.outcome_unknown and request.token_total is not None
            and await invoke(ctrl.store.rejected_source_is_recorded, job_id, request.fence))
        result = await invoke(ctrl.store.fail_job, job_id, worker_id=request.worker_id, fence=request.fence,
            idempotency_key=f'{job_id}-fence-{request.fence}-fail', reason=request.reason or request.termination or request.error or 'teacher_failed',
            evidence_ref=evidence_ref,
            usage_tokens=request.token_total if request.usage_known else None,
            retryable=retry_source, safe_to_retry=retry_source, outcome_unknown=request.outcome_unknown, usage_is_total=True,
            confirmed_call_tokens=([receipt.get('total_tokens') for receipt in request.usage_calls]
                                   if not request.usage_known else ()))
        await invoke(ctrl.settle_unknown_usage_policy)
        result = await invoke(ctrl.store.get_job, job_id)
        session = ctrl.sessions.get(job_id)
        if session and session.fence == request.fence and session.env.session is not None:
            await invoke(session.env.close)
            session.finished = True
        return {'job_id': job_id, 'status': result.state}
