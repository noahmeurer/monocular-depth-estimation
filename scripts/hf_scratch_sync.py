#!/usr/bin/env python3
"""Push/pull cluster scratch artifacts via a private Hugging Face model repo.

Remote layout mirrors paths under SCRATCH_ROOT (default /work/scratch/nmeurer),
e.g. outputs/baseline1  ->  outputs/baseline1/... on the Hub.

Requires HF_TOKEN (read+write) and HF_SCRATCH_REPO_ID (.env).
Pass --scratch-root to push/pull from a different user's scratch.

Examples:
  cp .env.example .env   # set HF_TOKEN and HF_SCRATCH_REPO_ID

  python scripts/hf_scratch_sync.py push outputs/baseline1
  python scripts/hf_scratch_sync.py push outputs/baseline1 --upload-mode large
  python scripts/hf_scratch_sync.py push outputs/baseline3/pseudo_labels_DA3-GIANT-1.1 \\
      --chunk-size 1000 --sleep-between-chunks 30
  python scripts/hf_scratch_sync.py pull outputs/baseline1
  python scripts/hf_scratch_sync.py list outputs/

Chunked push splits a folder into N batches of files and commits each batch
separately. The Hub layout stays identical to the local folder, so `pull`
naturally reassembles without any special unchunk step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    snapshot_download,
)
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import (
    DEFAULT_IGNORE_PATTERNS,
    filter_repo_objects,
)

DEFAULT_REPO_TYPE = "model"
DEFAULT_SCRATCH_ROOT = Path("/work/scratch/nmeurer")

EXTRA_IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.ipynb_checkpoints/**",
    "**/.nfs*",
    "**/slurm-*.out",
    "**/slurm-*.err",
]

IGNORE_PATTERNS = list(DEFAULT_IGNORE_PATTERNS) + EXTRA_IGNORE_PATTERNS

# Same threshold as huggingface_hub's upload_folder warning (see hf_api._prepare_upload_folder_additions).
LARGE_FOLDER_HINT_MIN_FILES = 30


def _count_files(directory: Path) -> int:
    if directory.is_file():
        return 1
    return sum(1 for path in directory.rglob("*") if path.is_file())


def _use_large_upload(local: Path, mode: str) -> bool:
    if mode == "large":
        return True
    if mode in {"small", "auto"}:
        return False
    raise ValueError(f"Unsupported upload mode: {mode}")


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit(
            "Missing HF_TOKEN. Create a token at https://huggingface.co/settings/tokens "
            "and export it, or run `hf auth login`."
        )
    return token


def _repo_id(explicit: str | None) -> str:
    repo_id = explicit or os.environ.get("HF_SCRATCH_REPO_ID")
    if not repo_id:
        sys.exit(
            "Missing repo id. Pass --repo-id or set HF_SCRATCH_REPO_ID "
            "(e.g. your-org/cil-monocular-artifacts)."
        )
    return repo_id


def _rel_paths(scratch_root_path: Path, paths: list[str]) -> list[str]:
    rels: list[str] = []
    for raw in paths:
        p = Path(raw)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(scratch_root_path.resolve())
            except ValueError as exc:
                raise SystemExit(
                    f"Path {p} is not under scratch root {scratch_root_path}"
                ) from exc
            rels.append(rel.as_posix())
        else:
            rels.append(p.as_posix().lstrip("/"))
    return rels


def _ensure_private_repo(api: HfApi, repo_id: str, repo_type: str, token: str) -> None:
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=True,
        exist_ok=True,
        token=token,
    )
    api.update_repo_settings(
        repo_id=repo_id,
        repo_type=repo_type,
        private=True,
        token=token,
    )


def _list_local_files(local: Path) -> list[str]:
    """Sorted POSIX relative paths under `local`, honoring IGNORE_PATTERNS."""
    all_files = sorted(
        p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()
    )
    return list(filter_repo_objects(all_files, ignore_patterns=IGNORE_PATTERNS))


def _remote_files_set(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    revision: str | None,
    token: str,
) -> set[str]:
    try:
        return set(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )
        )
    except RepositoryNotFoundError:
        return set()


def _upload_chunks(
    api: HfApi,
    *,
    repo_id: str,
    repo_type: str,
    token: str,
    revision: str | None,
    rel: str,
    local: Path,
    chunk_size: int,
    sleep_between_chunks: float,
    commit_message: str | None,
) -> None:
    relative_files = _list_local_files(local)
    total = len(relative_files)
    if total == 0:
        print(f"  Nothing to upload under {local}")
        return

    n_chunks = (total + chunk_size - 1) // chunk_size
    remote_files = _remote_files_set(
        api,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )

    print(
        f"Chunked upload: {total} files in {n_chunks} chunks "
        f"(chunk_size={chunk_size}, sleep_between_chunks={sleep_between_chunks:g}s)"
    )

    for idx in range(n_chunks):
        batch = relative_files[idx * chunk_size : (idx + 1) * chunk_size]
        remote_for_batch = {f"{rel}/{p}" for p in batch}

        if remote_for_batch.issubset(remote_files):
            print(f"  Chunk {idx + 1}/{n_chunks}: skip ({len(batch)} files already on Hub)")
            continue

        missing = sorted(remote_for_batch - remote_files)
        print(
            f"  Chunk {idx + 1}/{n_chunks}: commit {len(missing)} new / {len(batch)} files"
        )

        ops = [
            CommitOperationAdd(
                path_in_repo=f"{rel}/{p}",
                path_or_fileobj=str((local / p).resolve()),
            )
            for p in batch
            if f"{rel}/{p}" not in remote_files
        ]
        if not ops:
            continue

        msg = (
            commit_message
            or f"Sync scratch/{rel} chunk {idx + 1}/{n_chunks}"
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=msg,
            revision=revision,
            token=token,
        )
        remote_files.update(op.path_in_repo for op in ops)

        if idx + 1 < n_chunks and sleep_between_chunks > 0:
            time.sleep(sleep_between_chunks)


def _upload_one(
    api: HfApi,
    *,
    repo_id: str,
    scratch_root_path: Path,
    rel: str,
    local: Path,
    repo_type: str,
    token: str,
    revision: str | None,
    commit_message: str | None,
    upload_mode: str,
    num_workers: int | None,
    chunk_size: int,
    sleep_between_chunks: float,
) -> None:
    n_files = _count_files(local)
    if local.is_file():
        print(f"Single-file upload -> {repo_id}:{rel}")
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=local,
            path_in_repo=rel,
            commit_message=commit_message or f"Sync scratch/{rel}",
            token=token,
            revision=revision,
        )
        return

    if chunk_size > 0:
        _upload_chunks(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            revision=revision,
            rel=rel,
            local=local,
            chunk_size=chunk_size,
            sleep_between_chunks=sleep_between_chunks,
            commit_message=commit_message,
        )
        return

    use_large = _use_large_upload(local, upload_mode)

    if use_large:
        print(
            f"Resumable large upload ({n_files} files) -> {repo_id}:{rel}\n"
            "  (interrupt-safe; metadata in .cache/.huggingface under scratch root)"
        )
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=scratch_root_path,
            repo_type=repo_type,
            revision=revision,
            private=True,
            allow_patterns=[f"{rel}/**", rel],
            ignore_patterns=IGNORE_PATTERNS,
            num_workers=num_workers,
        )
        return

    if n_files > LARGE_FOLDER_HINT_MIN_FILES:
        print(
            f"Standard upload ({n_files} files). "
            "For thousands of files, prefer --chunk-size to avoid HF commit-rate limits."
        )

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=local,
        path_in_repo=rel,
        commit_message=commit_message or f"Sync scratch/{rel}",
        ignore_patterns=IGNORE_PATTERNS,
        token=token,
        revision=revision,
    )


def cmd_push(
    api: HfApi,
    *,
    repo_id: str,
    scratch_root_path: Path,
    rel_paths: list[str],
    repo_type: str,
    token: str,
    revision: str | None,
    commit_message: str | None,
    dry_run: bool,
    upload_mode: str,
    num_workers: int | None,
    chunk_size: int,
    sleep_between_chunks: float,
) -> None:
    _ensure_private_repo(api, repo_id, repo_type, token)

    for rel in rel_paths:
        local = scratch_root_path / rel
        if not local.exists():
            sys.exit(f"Local path does not exist: {local}")

        if local.is_file():
            mode_label = "file"
        elif chunk_size > 0:
            mode_label = f"chunked(chunk_size={chunk_size})"
        elif _use_large_upload(local, upload_mode):
            mode_label = "large"
        else:
            mode_label = "standard"

        print(
            f"{'[dry-run] ' if dry_run else ''}Upload ({mode_label}) "
            f"{local} -> {repo_id}:{rel} ({_count_files(local)} files)"
        )

        if dry_run:
            continue

        _upload_one(
            api,
            repo_id=repo_id,
            scratch_root_path=scratch_root_path,
            rel=rel,
            local=local,
            repo_type=repo_type,
            token=token,
            revision=revision,
            commit_message=commit_message,
            upload_mode=upload_mode,
            num_workers=num_workers,
            chunk_size=chunk_size,
            sleep_between_chunks=sleep_between_chunks,
        )


def cmd_pull(
    *,
    repo_id: str,
    scratch_root_path: Path,
    rel_paths: list[str],
    repo_type: str,
    token: str,
    revision: str | None,
    force_download: bool,
    dry_run: bool,
) -> None:
    scratch_root_path.mkdir(parents=True, exist_ok=True)

    for rel in rel_paths:
        patterns = [f"{rel}/**", rel] if rel != "." else None
        dest = scratch_root_path
        print(
            f"{'[dry-run] ' if dry_run else ''}Download {repo_id}:{rel} -> {dest / rel if rel != '.' else dest}"
        )

        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=dest,
            allow_patterns=patterns,
            token=token,
            force_download=force_download,
            dry_run=dry_run,
        )


def cmd_list(
    api: HfApi,
    *,
    repo_id: str,
    prefix: str | None,
    repo_type: str,
    token: str,
    revision: str | None,
) -> None:
    try:
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    except RepositoryNotFoundError:
        print(
            f"Repository {repo_id} does not exist yet (or you lack access).\n"
            "Run a push first to create the private repo, e.g.:\n"
            "  python scripts/hf_scratch_sync.py push outputs/baseline1"
        )
        return

    if prefix:
        prefix = prefix.strip("/")
        if prefix:
            prefix = prefix + "/"
            files = [f for f in files if f == prefix.rstrip("/") or f.startswith(prefix)]

    if not files:
        print("(no files)")
        return
    for path in sorted(files):
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync scratch directories with a private Hugging Face model repo.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo-id",
        help="Hub repo id namespace/name (default: HF_SCRATCH_REPO_ID)",
    )
    common.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=("model", "dataset"),
        help=f"Hub repo type (default: {DEFAULT_REPO_TYPE})",
    )
    common.add_argument(
        "--revision",
        default=None,
        help="Git revision (branch, tag, or commit) to read/write",
    )
    common.add_argument(
        "--scratch-root",
        default=str(DEFAULT_SCRATCH_ROOT),
        help=f"Local scratch root (default: {DEFAULT_SCRATCH_ROOT})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    push_p = sub.add_parser(
        "push",
        parents=[common],
        help="Upload scratch paths to the Hub (private repo)",
    )
    push_p.add_argument(
        "paths",
        nargs="+",
        help="Path(s) relative to scratch, e.g. outputs/baseline1",
    )
    push_p.add_argument(
        "-m",
        "--message",
        help="Commit message (default: per-path auto message)",
    )
    push_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without uploading",
    )
    push_p.add_argument(
        "--upload-mode",
        choices=("auto", "large", "small"),
        default="auto",
        help=(
            "Upload strategy: auto defaults to standard upload_folder "
            "(safer for HF commit-rate limits); use large for resumable uploads"
        ),
    )
    push_p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel workers for large uploads (default: half of CPU cores)",
    )
    push_p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Split the folder into chunks of N files; commit each chunk separately. "
            "Hub layout is unchanged, so pull reassembles automatically. "
            "Already-uploaded files are skipped via the remote file listing."
        ),
    )
    push_p.add_argument(
        "--sleep-between-chunks",
        type=float,
        default=30.0,
        metavar="S",
        help=(
            "Seconds to sleep between chunk commits (default: 30). "
            "HF free plans allow ~128 commits/hour; 30s/commit keeps you safely under."
        ),
    )

    pull_p = sub.add_parser(
        "pull",
        parents=[common],
        help="Download Hub paths into scratch",
    )
    pull_p.add_argument(
        "paths",
        nargs="+",
        help="Path(s) relative to scratch, e.g. outputs/baseline1",
    )
    pull_p.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally",
    )
    pull_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading",
    )

    list_p = sub.add_parser(
        "list",
        parents=[common],
        help="List remote files (optional prefix filter)",
    )
    list_p.add_argument(
        "prefix",
        nargs="?",
        default=None,
        help="Only list files under this prefix, e.g. outputs/",
    )

    args = parser.parse_args()
    token = _token()
    repo_id = _repo_id(args.repo_id)
    scratch_root_path = Path(args.scratch_root).expanduser().resolve()
    api = HfApi(token=token)

    if args.command == "list":
        cmd_list(
            api,
            repo_id=repo_id,
            prefix=args.prefix,
            repo_type=args.repo_type,
            token=token,
            revision=args.revision,
        )
        return

    rel_paths = _rel_paths(scratch_root_path, args.paths)

    if args.command == "push":
        cmd_push(
            api,
            repo_id=repo_id,
            scratch_root_path=scratch_root_path,
            rel_paths=rel_paths,
            repo_type=args.repo_type,
            token=token,
            revision=args.revision,
            commit_message=args.message,
            dry_run=getattr(args, "dry_run", False),
            upload_mode=args.upload_mode,
            num_workers=args.num_workers,
            chunk_size=max(args.chunk_size, 0),
            sleep_between_chunks=max(args.sleep_between_chunks, 0.0),
        )
    elif args.command == "pull":
        cmd_pull(
            repo_id=repo_id,
            scratch_root_path=scratch_root_path,
            rel_paths=rel_paths,
            repo_type=args.repo_type,
            token=token,
            revision=args.revision,
            force_download=args.force,
            dry_run=getattr(args, "dry_run", False),
        )


if __name__ == "__main__":
    main()
