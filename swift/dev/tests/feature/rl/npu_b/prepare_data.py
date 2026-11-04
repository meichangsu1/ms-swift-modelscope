#!/usr/bin/env python3
"""Prepare local Parquet snapshots for the B-owned GRPO cases, without network access.

Exports embedded images/videos and writes prompt-only JSONL with the solution retained.
Only pyarrow and Pillow are required. Audio in MTEB/Video-MME_short is not read.
"""

import argparse
import hashlib
import io
import re
from pathlib import Path

import json

SPECS = {
    'gsm8k': ('main/train*.parquet', ('question', 'answer')),
    'numina_math_tir': ('data/train*.parquet', ('problem', 'solution')),
    'single_image': ('data/train*.parquet', ('problem', 'solution', 'image')),
    'multi_image': ('data/test*.parquet', ('idx', 'question', 'options', 'answer', 'image_list')),
    'short_video': ('data/test*.parquet', ('question_id', 'question', 'candidates', 'answer', 'video')),
}


def find_shards(source, kind):
    if source.is_file():
        return [source]
    pattern = SPECS[kind][0]
    shards = sorted(source.glob(pattern))
    if not shards:
        shards = sorted(source.glob(Path(pattern).name))
    if not shards:
        raise FileNotFoundError(f'No {pattern} files under {source}. Set the corresponding dataset directory.')
    return shards


def read_rows(shards, columns):
    import pyarrow.parquet as pq

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f'{shard} is missing columns: {sorted(missing)}')
        # Keep embedded video memory bounded and skip the unrelated audio column entirely.
        for batch in parquet.iter_batches(batch_size=1, columns=list(columns)):
            yield batch.to_pylist()[0]


def media_path(value, source, media_dir, kind):
    if isinstance(value, str):
        data, original = None, value
    elif isinstance(value, dict):
        data, original = value.get('bytes'), value.get('path')
    else:
        raise TypeError(f'Unsupported {kind} value: {type(value).__name__}')
    if data is None:
        original = str(original or '')
        if not original or '://' in original:
            raise ValueError(f'{kind} has no embedded bytes or local path: {original!r}')
        path = Path(original)
        if not path.is_absolute():
            path = source / path
        if not path.is_file():
            raise FileNotFoundError(f'Local {kind} not found: {path}')
        return str(path.resolve())

    digest = hashlib.sha256(data).hexdigest()
    if kind == 'image':
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            suffix = '.' + image.format.lower()
    else:
        suffix = Path(original or '').suffix.lower() or '.mp4'
    path = media_dir / (digest + suffix)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
    return str(path.resolve())


def mcq_prompt(question, options):
    labelled = []
    for index, option in enumerate(options):
        text = str(option)
        if not re.match(r'^\s*[A-Z][.)：:]\s*', text):
            text = f'{chr(65 + index)}. {text}'
        labelled.append(text)
    return str(question).strip() + '\n' + '\n'.join(labelled)


def mcq_solution(answer):
    match = re.match(r'^\s*([A-Z])(?:[.)：:]|\s|$)', str(answer))
    if match is None:
        raise ValueError(f'Expected an option letter or labelled option; found {answer!r}')
    # MathAccuracy parses LaTeX; a bare option letter may not be extracted as an expression.
    return '\\boxed{' + match.group(1) + '}'


def convert_row(row, kind, source, media_dir, index):
    result = {'sample_id': f'{kind}:{row.get("idx", row.get("question_id", index))}'}
    if kind == 'gsm8k':
        question = str(row['question'])
        result['solution'] = str(row['answer']).rsplit('####', 1)[-1].strip()
    elif kind in ('numina_math_tir', 'single_image'):
        question = str(row['problem'])
        result['solution'] = str(row['solution'])
    else:
        options = row['options'] if kind == 'multi_image' else row['candidates']
        question = mcq_prompt(row['question'], options)
        result['solution'] = mcq_solution(row['answer'])

    if kind == 'single_image':
        result['images'] = [media_path(row['image'], source, media_dir, 'image')]
    elif kind == 'multi_image':
        images = row['image_list']
        if len(images) != 2:
            return None
        result['images'] = [media_path(image, source, media_dir, 'image') for image in images]
    elif kind == 'short_video':
        result['videos'] = [media_path(row['video'], source, media_dir, 'video')]

    for column, marker in (('images', '<image>'), ('videos', '<video>')):
        count = len(result.get(column, []))
        present = question.count(marker)
        if count and not present:
            question = (marker + '\n') * count + question
        elif present != count:
            raise ValueError(f'Media placeholder mismatch in {result["sample_id"]}: {present} vs {count}')
    result['messages'] = [{'role': 'user', 'content': question}]
    return result


def cache_ready(output, manifest_path, expected):
    if not output.is_file() or not manifest_path.is_file():
        return False
    actual = json.loads(manifest_path.read_text(encoding='utf-8'))
    if actual.get('inputs') != expected:
        return False
    for line in output.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        for column in ('images', 'videos'):
            if any(not Path(path).is_file() for path in row.get(column, [])):
                return False
    return output.stat().st_size == actual.get('output_size')


def prepare(args):
    source = args.source.expanduser().resolve()
    shards = find_shards(source, args.dataset)
    output_dir = args.output_dir.expanduser().resolve()
    output = output_dir / f'{args.dataset}_{args.limit}.jsonl'
    manifest_path = output.with_suffix('.manifest.json')
    inputs = {'version': 1, 'dataset': args.dataset, 'limit': args.limit, 'sources': [
        {'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in shards
    ]}
    if cache_ready(output, manifest_path, inputs) and not args.refresh:
        print(f'Reusing {output}', flush=True)
        return output

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source.parent if source.is_file() else source
    media_dir = output_dir / 'media' / args.dataset
    count = 0
    temporary = output.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for index, row in enumerate(read_rows(shards, SPECS[args.dataset][1])):
            converted = convert_row(row, args.dataset, source_dir, media_dir, index)
            if converted is None:
                continue
            stream.write(json.dumps(converted, ensure_ascii=False) + '\n')
            count += 1
            if count == args.limit:
                break
    if count < args.limit:
        temporary.unlink()
        raise ValueError(f'{args.dataset}: need {args.limit} eligible rows; found {count} in {source}.')
    temporary.replace(output)
    manifest_path.write_text(json.dumps({'inputs': inputs, 'rows': count, 'output_size': output.stat().st_size},
                                        indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {count} rows: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=SPECS, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=256)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
