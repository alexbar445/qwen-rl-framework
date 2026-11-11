from qwen_rl.contracts.sampling_topology import owner_map, owner_slots
from qwen_rl.worker.episode_queue import EpisodeRoleQueue
from qwen_rl.learner.gradient_server.store import SQLiteGradientJobStore


def topology():
    return {
        'gradient_executor_worker_ids': {f'executor-{i}': f'worker-{i}' for i in range(4)},
        'sampling_only_worker_ids': {f'sampler-{i}': f'wan-{i}' for i in range(4)},
        'gradient_tasks_per_worker': 2,
        'sampling_group_quotas': {},
    }


def test_twelve_slots_preserve_owner_and_quota():
    learner = topology()
    slots = owner_slots(learner)
    workers = owner_map(learner)
    assert len(slots) == 12
    assert {workers[owner]: slots.count(owner) for owner in slots} == {
        **{f'worker-{i}': 2 for i in range(4)}, **{f'wan-{i}': 1 for i in range(4)}
    }


def test_external_owner_waits_for_policy_and_cannot_use_other_owner_gpu():
    queue = EpisodeRoleQueue({'worker-0': 'cloud', 'wan-0': 'wan'},
                             {'worker-0': 2, 'wan-0': 2}, external_owner_ids={'wan-0'})
    policy = (3, 'sha')
    queue.enqueue('one', 'wan-0', policy, 'task')
    assert queue.claim_episode('one', {'worker-0', 'wan-0'}) is None
    queue.set_helper_policy(policy)
    assert queue.claim_episode('one', {'worker-0', 'wan-0'}) == 'wan-0'


def test_sampling_round_counts_heterogeneous_slots(tmp_path):
    learner = topology()
    slots = owner_slots(learner)
    store = SQLiteGradientJobStore(tmp_path / 'jobs.db', tmp_path / 'rows')
    store.begin_sampling_round('r', lineage_id='lineage', engine_epoch='epoch', policy_version=0,
                               executor_ids=tuple(sorted(owner_map(learner))), tasks_per_worker=2,
                               first_ordinal=0, owner_slots=slots)
    for i in range(12):
        store.note_sampling_outcome('r', i)
    outcomes = store._sampling_rounds['r']['outcomes']
    assert len(outcomes) == 12
    assert [outcomes[i] for i in range(12)] == list(slots)


def test_cloud_gpu_can_compute_while_external_owner_is_generating():
    queue = EpisodeRoleQueue({'worker-0': 'cloud', 'wan-0': 'wan'},
                             {'worker-0': 1, 'wan-0': 1}, external_owner_ids={'wan-0'})
    policy = (3, 'sha')
    queue.set_helper_policy(policy)
    queue.enqueue('slow', 'wan-0', policy, 'task')
    assert queue.claim_episode('slow', {'wan-0'}) == 'wan-0'
    with queue.compute_admission('worker-0', 'learner-session') as admitted:
        assert admitted
    with queue.compute_admission('wan-0', 'other-session') as admitted:
        assert not admitted


def test_managed_pair_renaming_preserves_explicit_quotas():
    from qwen_rl.contracts.sampling_topology import remap_paired_quotas
    learner = {'gradient_executor_worker_ids': {'e0': 'old-a', 'e1': 'old-b'},
               'sampling_group_quotas': {'old-a': 3, 'wan': 1}}
    assert remap_paired_quotas(learner, {'e0': 'worker-0', 'e1': 'worker-1'}) == {
        'worker-0': 3, 'wan': 1,
    }


def shared_topology():
    learner = topology()
    learner['sampling_shared_worker_pairs'] = {'wan-0':'wan-1', 'wan-2':'wan-3'}
    learner['sampling_episode_slots'] = {f'wan-{i}':3 for i in range(4)}
    return learner


def test_shared_pairs_rotate_two_group_owners_and_keep_cloud_quotas():
    learner=shared_topology()
    for version, wanted in ((196, {'sampler-0','sampler-2'}),
                            (197, {'sampler-1','sampler-3'})):
        slots=owner_slots(learner,rotation=version)
        assert len(slots)==10
        assert {x for x in slots if x.startswith('sampler-')}==wanted
        assert all(slots.count(f'executor-{i}')==2 for i in range(4))
        assert set(owner_map(learner).values()) >= {f'wan-{i}' for i in range(4)}


def test_two_complete_six_trajectory_groups_share_four_gpus():
    pools={'wan-0':'pair-0','wan-1':'pair-0','wan-2':'pair-1','wan-3':'pair-1'}
    for version, owners in ((196,('wan-0','wan-2')), (197,('wan-1','wan-3'))):
        queue=EpisodeRoleQueue(pools,dict.fromkeys(pools,3),external_owner_ids=set(pools))
        policy=(version,'sha');queue.set_helper_policy(policy)
        for owner in owners:
            for i in range(6):queue.enqueue(f'{owner}-{i}',owner,policy,'task')
        assigned={worker:0 for worker in pools}
        for owner in owners:
            for i in range(6):
                target=queue.claim_episode(f'{owner}-{i}',set(pools))
                assert target is not None and pools[target]==pools[owner]
                assigned[target]+=1
        assert assigned==dict.fromkeys(pools,3)


def test_shared_sampling_round_attests_only_actual_owner_slots(tmp_path):
    learner=shared_topology();store=SQLiteGradientJobStore(tmp_path/'jobs.db',tmp_path/'rows')
    for version in (196,197):
        slots=owner_slots(learner,rotation=version)
        store.begin_sampling_round(str(version),lineage_id='lineage',engine_epoch='epoch',
            policy_version=version,executor_ids=tuple(sorted(set(slots))),tasks_per_worker=2,
            first_ordinal=0,owner_slots=slots)
        for i in range(10):store.note_sampling_outcome(str(version),i)
        assert list(store._sampling_rounds[str(version)]['outcomes'].values())==list(slots)


def test_bepa_extra_row_waits_for_a_real_slot_and_then_runs():
    pools={'wan-0':'pair','wan-1':'pair'}
    queue=EpisodeRoleQueue(pools,dict.fromkeys(pools,3),external_owner_ids=set(pools))
    policy=(196,'sha');queue.set_helper_policy(policy)
    for i in range(7):queue.enqueue(f'row-{i}','wan-0',policy,'task',allow_help=(i != 6))
    targets=[queue.claim_episode(f'row-{i}',set(pools)) for i in range(6)]
    assert targets.count('wan-0')==3 and targets.count('wan-1')==3
    assert queue.claim_episode('row-6',set(pools)) is None
    queue.note_generation_finished('row-0')
    assert queue.claim_episode('row-6',set(pools))=='wan-0'


def test_invalid_shared_pairs_are_rejected():
    import pytest
    for pairs in ({'wan-0':'wan-0'}, {'wan-0':'wan-1','wan-2':'wan-1'},
                  {'wan-0':'worker-0'}, {'wan-0':'missing'}):
        learner=topology();learner['sampling_shared_worker_pairs']=pairs
        with pytest.raises(ValueError, match='shared sampling pairs'):
            owner_slots(learner)
