"""Oracle queue adapter for the complete forecast lifecycle producer.

Local callers submit immutable source inputs. Only the queue worker may enter the
native producer; a backend failure never starts local TLC.
"""
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

PROJECT = 'sloik/superforecasting'
OUTPUT = '.tla-queue-output'


def is_worker():
    marker = os.environ.get('TLA_QUEUE_JOB')
    if not marker:
        return False
    if (not re.fullmatch('[0-9a-f]{32}', marker)
            or sys.platform != 'linux'
            or pwd.getpwuid(os.getuid()).pw_name != 'tla-job'):
        raise RuntimeError('invalid Oracle queue worker context; no local TLC execution')
    cgroup = Path('/proc/self/cgroup').read_text()
    if not re.search(r'(?:^|/)tla-q-' + re.escape(marker) + r'\.service(?:/|$)', cgroup, re.MULTILINE):
        raise RuntimeError('Oracle queue job cgroup mismatch; no local TLC execution')
    return True


def _read_owned_logs(source, names, require_complete):
    """Fetch was restored by the queue client; still reject links and unknown names."""
    values = {}
    if source.is_symlink():
        raise RuntimeError('fetched output directory must not be a symlink')
    if not source.exists():
        if require_complete:
            raise RuntimeError('queue completed without fetched native logs')
        return values
    root_fd = os.open(str(source), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in names:
            try:
                st = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                if require_complete:
                    raise RuntimeError('missing native log: ' + name)
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise RuntimeError('native log must be a regular unlinked file: ' + name)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            with os.fdopen(fd, 'rb') as stream:
                values[name] = stream.read()
    finally:
        os.close(root_fd)
    return values


def merge_logs(source, destination, names, require_complete=True):
    """Atomically replace only owned logs; keep unrelated caller files intact."""
    if any('/' in name or name in ('.', '..') for name in names):
        raise RuntimeError('invalid owned log name')
    values = _read_owned_logs(Path(source), names, require_complete)
    if not values and not require_complete:
        return
    destination = Path(destination)
    if destination.is_symlink():
        raise RuntimeError('output directory must not be a symlink')
    destination.mkdir(parents=True, exist_ok=True)
    directory = os.open(str(destination), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Validate every current target before changing any owned file.
        for name in values:
            try:
                st = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise RuntimeError('output log target must be regular and unlinked: ' + name)
        for name, data in values.items():
            temporary = '.forecast-log-' + uuid.uuid4().hex
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
        os.fsync(directory)
    finally:
        os.close(directory)


def run(args, root, cases):
    client = shutil.which('tla-queue')
    if client is None:
        print('FAIL: Oracle queue client tla-queue unavailable; no local TLC fallback', file=sys.stderr)
        return 78
    with tempfile.TemporaryDirectory(prefix='forecast-queue-') as temporary:
        fetched = Path(temporary) / 'fetched'
        source = Path(temporary) / 'source'
        # Copy only the actual producer closure, not old proof outputs, bytecode,
        # Git metadata, credentials or unrelated forecasting ledger inputs.
        paths = ['tools/run_formal.py', 'tools/remote_formal.py', 'formal/ForecastLifecycle.tla']
        paths += ['formal/' + case + '.cfg' for case, _ in cases]
        root_fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for path in paths:
                parent, name = path.split('/')
                directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                    with os.fdopen(fd, 'rb') as original:
                        st = os.fstat(original.fileno())
                        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                            raise OSError('producer input must be a regular unlinked file: ' + path)
                        copied = source / path
                        copied.parent.mkdir(parents=True, exist_ok=True)
                        with copied.open('xb') as destination:
                            shutil.copyfileobj(original, destination)
                finally:
                    os.close(directory)
        except OSError as exc:
            print('FAIL: producer input snapshot: ' + str(exc), file=sys.stderr)
            return 78
        finally:
            os.close(root_fd)
        command = [client, 'run', '--project', PROJECT, '--source', str(source),
                   '--include', 'tools', '--include', 'formal', '--timeout', '900',
                   '--output', OUTPUT]
        if args.output:
            command += ['--fetch', str(fetched)]
        command += ['--', '/usr/bin/python3', '-B', 'tools/run_formal.py',
                    '--java', args.java, '--jar', str(args.jar), '--output', OUTPUT]
        try:
            result = subprocess.run(command, cwd=root)
        except OSError as exc:
            print('FAIL: Oracle queue backend unavailable: ' + str(exc) + '; no local TLC fallback', file=sys.stderr)
            return 75
        if args.output:
            try:
                merge_logs(fetched / OUTPUT, args.output,
                           [case + '.log' for case, _ in cases],
                           require_complete=result.returncode == 0)
            except (OSError, RuntimeError) as exc:
                print('FAIL: remote native-log retrieval: ' + str(exc), file=sys.stderr)
                return 1 if result.returncode == 0 else result.returncode
        return result.returncode
