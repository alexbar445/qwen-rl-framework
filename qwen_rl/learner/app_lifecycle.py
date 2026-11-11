"""Learner app startup and shutdown lifecycle."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI

from ..rollout.groups import PolicySnapshot


def make_learner_lifespan(
    app: FastAPI,
    *,
    engine: Any,
    recovered_gradient_step_callback: Callable[[Any], Callable[[], None] | None],
    publish_current: Callable[..., Any],
    sync_workers: Callable[[dict[str, Any]], Any],
    latest_policy: Callable[[], dict[str, Any]],
):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            resume_gradient = getattr(engine, "resume_ready_gradient_windows", None)
            if callable(resume_gradient):
                durable_version_before_resume = int(engine.policy_version)
                await resume_gradient(callback_factory=recovered_gradient_step_callback)
                if int(engine.policy_version) != durable_version_before_resume:
                    if int(engine.policy_version) not in app.state.policies:
                        await publish_current()
                    await sync_workers(latest_policy())
                if getattr(app.state, "bepa", None) is not None:
                    app.state.bepa.recovery_metadata = engine.recovery_metadata
            if not app.state.initial_policy_ready:
                await publish_current()
                app.state.initial_policy_ready = True
            policy = app.state.policies.get(int(engine.policy_version))
            if policy is None:
                raise RuntimeError("recovery startup requires the current policy to be published")
            if app.state.bepa is not None:
                app.state.recovery_resolution = await asyncio.to_thread(
                    app.state.bepa.recover_startup,
                    policy_version=int(engine.policy_version),
                    adapter_sha256=PolicySnapshot.from_policy(policy).adapter_sha256,
                )
            stop_control = getattr(app.state, "stop_control", None)
            reconcile_stop = getattr(stop_control, "reconcile_startup", None)
            if callable(reconcile_stop):
                reconcile_stop(
                    recovery=getattr(engine, "recovery_metadata", None),
                    bepa=app.state.bepa,
                )
            app.state.recovery_initialized = True
            inference_pool = getattr(app.state, 'inference_pool', None)
            if inference_pool is not None:
                inference_pool.request_sync()
        except BaseException:
            inference_pool = getattr(app.state, 'inference_pool', None)
            if inference_pool is not None:
                await inference_pool.close()
            worker_control = getattr(app.state, 'worker_control', None)
            if worker_control is not None:
                await asyncio.to_thread(worker_control.close)
            raise
        try:
            yield
        finally:
            if app.state.evaluation_coordinator is not None:
                await app.state.evaluation_coordinator.close()
            inference_pool = getattr(app.state, 'inference_pool', None)
            if inference_pool is not None:
                await inference_pool.close()
            request_shutdown = getattr(engine, "request_shutdown", None)
            if callable(request_shutdown):
                request_shutdown()
            if app.state.active_task is not None and not app.state.active_task.done():
                app.state.active_task.cancel()
                await asyncio.gather(app.state.active_task, return_exceptions=True)
            worker_control = getattr(app.state, 'worker_control', None)
            if worker_control is not None:
                await asyncio.to_thread(worker_control.close)
            if app.state.bepa is not None:
                await asyncio.to_thread(app.state.bepa.close)
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                await asyncio.to_thread(shutdown)

    return lifespan
