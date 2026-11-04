#!/usr/bin/env python3
"""Download the NPU GRPO test datasets on a networked preparation machine.

Only huggingface_hub is required; this script does not import Swift or Torch.
Examples (run from the repository root):
    python swift/dev/tests/feature/rl/download_npu_grpo_datasets.py --output-dir /data/npu_grpo
    python swift/dev/tests/feature/rl/download_npu_grpo_datasets.py --output-dir /data/npu_grpo --include-videos
    python swift/dev/tests/feature/rl/download_npu_grpo_datasets.py --datasets video_mme --include-videos --dry-run

The first four repositories are downloaded in full, including their image data.
Video-MME defaults to annotations only. --include-videos adds the video ZIPs,
about 101 GB; extraction and training-format conversion are separate steps.
Re-running against the same output directory reuses Hugging Face download metadata.
"""

import argparse
import fnmatch
import json
from pathlib import Path


DATASETS = {
    'gsm8k': ('openai/gsm8k', 'main', 'train'),
    'numina_math_tir': ('project-numina/NuminaMath-TIR', 'default', 'train'),
    'multimodal_open_r1': ('lmms-lab/multimodal-open-r1-8k-verified', 'default', 'train'),
    'muirbench': ('MUIRBENCH/MUIRBENCH', 'default', 'test'),
    'video_mme': ('lmms-eval/Video-MME', 'videomme', 'test'),
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=Path('data/npu_grpo'))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--include-videos', action='store_true',
                        help='Also download Video-MME video ZIPs (about 101 GB).')
    parser.add_argument('--max-workers', type=positive_int, default=4)
    parser.add_argument('--dry-run', action='store_true',
                        help='Query file lists and sizes without downloading dataset files.')
    args = parser.parse_args()
    if args.include_videos and 'video_mme' not in args.datasets:
        parser.error('--include-videos requires video_mme in --datasets')
    return args


def download_patterns(key, include_videos):
    if key != 'video_mme':
        return None
    patterns = ['README.md', 'videomme/*.parquet']
    if include_videos:
        patterns.append('videos_chunked_*.zip')
    return patterns


def select_files(repo_id, siblings, patterns, include_videos):
    files = [file for file in siblings
             if patterns is None or any(fnmatch.fnmatchcase(file.rfilename, pattern) for pattern in patterns)]
    required = ['*.parquet']
    if repo_id == DATASETS['video_mme'][0] and include_videos:
        required.append('videos_chunked_*.zip')
    for pattern in required:
        if not any(fnmatch.fnmatchcase(file.rfilename, pattern) for file in files):
            raise RuntimeError(f'{repo_id}: no files matched required pattern {pattern}.')
    return files


def verify_files(destination, files):
    # Check the portable local files, rather than trusting only the cache markers.
    for file in files:
        path = destination / file.rfilename
        if not path.is_file() or (file.size is not None and path.stat().st_size != file.size):
            raise RuntimeError(f'Downloaded file is missing or has the wrong size: {path}')


def download_dataset(args, api, snapshot_download, key, root):
    repo_id, config, split = DATASETS[key]
    patterns = download_patterns(key, args.include_videos)
    # Resolve the commit once so every file in this download uses the same revision.
    info = api.dataset_info(repo_id, files_metadata=True)
    files = select_files(repo_id, info.siblings, patterns, args.include_videos)
    destination = root / repo_id
    size = sum(file.size or 0 for file in files)
    unknown_sizes = sum(file.size is None for file in files)
    size_label = f'{size / 1_000_000_000:.3f} GB'
    if unknown_sizes:
        size_label += f' plus {unknown_sizes} files with unknown size'
    print(f'\n{repo_id}@{info.sha}\n  {len(files)} files, {size_label}\n  -> {destination}', flush=True)
    if args.dry_run:
        return None

    snapshot_download(
        repo_id=repo_id,
        repo_type='dataset',
        revision=info.sha,
        local_dir=destination,
        allow_patterns=patterns,
        max_workers=args.max_workers,
    )
    verify_files(destination, files)
    scope = 'full_repository'
    if key == 'video_mme':
        scope = 'annotations_and_video_zips' if args.include_videos else 'annotations_only'
    return {
        'repo_id': repo_id,
        'revision': info.sha,
        'local_dir': repo_id,
        'test_config': config,
        'test_split': split,
        'download_scope': scope,
        'files': [{'path': file.rfilename, 'size': file.size} for file in files],
    }


def download(args):
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        raise SystemExit('Missing huggingface_hub. Install it on the preparation machine: '
                         'python -m pip install "huggingface_hub>=0.23"') from None

    root = args.output_dir.expanduser().resolve()
    manifest_path = root / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    api = HfApi()
    for key in dict.fromkeys(args.datasets):
        record = download_dataset(args, api, snapshot_download, key, root)
        if record is None:
            continue
        manifest[key] = record
        # Keep completed entries if a later dataset download is interrupted.
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(manifest_path)

    if args.dry_run:
        print('\nDry run finished; no dataset files were downloaded.')
    else:
        print(f'\nDownload manifest: {manifest_path}\nCopy the entire output directory to the offline machine.')
    if 'video_mme' in args.datasets and not args.include_videos:
        print('Video-MME is annotations-only in this run; video test data is not complete. '
              'Use --datasets video_mme --include-videos to download the video ZIPs.')
    elif 'video_mme' in args.datasets:
        print('Video ZIPs must be extracted and mapped to local video paths before training.')


if __name__ == '__main__':
    download(parse_args())
