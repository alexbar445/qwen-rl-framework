"""Offline pilot CLI; never sends requests to the running Learner or provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ready', required=True, type=Path)
    parser.add_argument('--message-id', required=True, type=int)
    parser.add_argument('--source-text-sha256', required=True)
    parser.add_argument('--strategy', choices=['observed-facts', 'grid-rows'], default='observed-facts')
    parser.add_argument('--insert-at', type=int)
    parser.add_argument('--replace-until', type=int)
    parser.add_argument('--witness-call-index', type=int, default=0)
    parser.add_argument('--solutions', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.strategy == 'observed-facts' and (args.insert_at is None or args.replace_until is None):
        parser.error('observed-facts requires --insert-at and --replace-until')
    result = run_pilot(**vars(args))
    print(json.dumps({k: result[k] for k in ('status', 'replay_equivalent', 'exact',
                                           'reasoning_preservation', 'cost', 'rejection_reasons')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
