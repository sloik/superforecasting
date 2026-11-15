"""Bounded binary lifecycle correspondence and formal CI controls."""
from __future__ import annotations
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from argparse import Namespace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tools'), str(Path(__file__).parent)]
from _ledger_guard import setUpModule, tearDownModule  # noqa: F401,E402
import forecastlog
from forecast_common import fold_forecast_log
import oracle_job_guard
import run_formal
import remote_formal


class LifecycleConformanceTests(unittest.TestCase):
    def test_all_two_id_four_event_binary_traces(self):
        visits = 0
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            log = Path(tmp) / 'ledger.jsonl'
            log.write_bytes(b'')

            def walk(expected, depth):
                nonlocal visits
                visits += 1
                before = log.read_bytes()
                actual = fold_forecast_log(str(log))
                self.assertEqual([r['id'] for r in actual], list(range(1, len(expected) + 1)))
                for row, wanted in zip(actual, expected):
                    self.assertEqual(row['status'], wanted['status'])
                    self.assertEqual(row['prob'], wanted['history'][-1])
                    history = [h['prob'] for h in row.get('prob_history', [{'prob': row['prob']}])]
                    self.assertEqual(history, wanted['history'])
                    if wanted['status'] == 'resolved':
                        self.assertEqual(row['outcome'], wanted['outcome'])
                        self.assertEqual(row['brier'], (wanted['history'][-1] - wanted['outcome']) ** 2)
                    else:
                        self.assertNotIn('brier', row)
                if depth == 4:
                    return
                commands = [('add', len(expected)+1, p) for p in (0.0, 0.5, 1.0)] if len(expected) < 2 else []
                for fid in (1, 2):
                    commands.extend(('update', fid, p) for p in (0.0, 0.5, 1.0))
                    commands.extend(('resolve', fid, o) for o in (0, 1))
                    commands.append(('supersede', fid, None))
                for action, fid, value in commands:
                    log.write_bytes(before)
                    following = copy.deepcopy(expected)
                    accepted = action == 'add' or (fid <= len(expected) and expected[fid-1]['status'] == 'open')
                    args = Namespace(log=str(log), id=fid, prob=value, outcome=value,
                                     outcome_bin=None, note='formal', rationale=None,
                                     reason='withdrawn', replaced_by=None)
                    if action == 'add':
                        args = Namespace(log=str(log), prob=value, dist=None, bins=None,
                                         question='bounded fixture', resolve_by=None, rationale=None,
                                         tags=None, forecaster=None, baseline_prob=None, crowd_prob=None,
                                         ensemble_prob=None, base_rate_prob=None)
                        following.append({'status': 'open', 'history': [value]})
                    elif accepted:
                        if action == 'update':
                            following[fid-1]['history'].append(value)
                        else:
                            following[fid-1]['status'] = 'resolved' if action == 'resolve' else 'superseded'
                            if action == 'resolve':
                                following[fid-1]['outcome'] = value
                    code = getattr(forecastlog, 'cmd_' + action)(args)
                    self.assertEqual(code, 0 if accepted else 1, (action, fid, value, expected))
                    after = log.read_bytes()
                    if accepted:
                        self.assertTrue(after.startswith(before))
                        self.assertEqual(len(after.splitlines()), depth + 1)
                        event = json.loads(after.splitlines()[-1])
                        self.assertEqual(event['id'], fid)
                        self.assertEqual(event['event'], 'forecast' if action == 'add' else action)
                        if action == 'resolve':
                            self.assertEqual(event['prob'], expected[fid-1]['history'][-1])
                            self.assertEqual(event['brier'], (expected[fid-1]['history'][-1] - value) ** 2)
                        elif action in {'add', 'update'}:
                            self.assertEqual(event['prob'], value)
                        walk(following, depth + 1)
                    else:
                        self.assertEqual(after, before)
                log.write_bytes(before)
            walk([], 0)
        self.assertEqual(visits, 2029)


class CorrespondenceControlTests(unittest.TestCase):
    def test_correspondence_detects_a_runtime_score_defect(self):
        with mock.patch.object(forecastlog, 'brier_score', lambda p, o: (0.5-o)**2):
            with self.assertRaises(AssertionError):
                LifecycleConformanceTests('test_all_two_id_four_event_binary_traces').test_all_two_id_four_event_binary_traces()


class FormalResultQualificationTests(unittest.TestCase):
    def test_only_complete_safe_results_qualify(self):
        text = 'Model checking completed. No error has been found.\n2029 distinct states found\n'
        self.assertTrue(run_formal.qualify(text, 0))
        for code in (1, 12, False, 0.0):
            self.assertFalse(run_formal.qualify(text, code))
        self.assertFalse(run_formal.qualify(text + 'Error: tool failed\n', 0))
        self.assertFalse(run_formal.qualify('No error has been found.', 0))

    def test_counterexample_requires_named_semantic_invariant(self):
        text = ('Error: Invariant HistoryComplete is violated.\n'
                'Error: The behavior up to this point is:\nState 1:\n3 distinct states found\n')
        self.assertTrue(run_formal.qualify(text, 12, 'HistoryComplete'))
        for wrong in ('TypeOK', 'TerminalImmutable'):
            self.assertFalse(run_formal.qualify(text, 12, wrong))
        for code in (0, 1, False, 12.0):
            self.assertFalse(run_formal.qualify(text, code, 'HistoryComplete'))
        self.assertFalse(run_formal.qualify(text + 'Error: tool failed\n', 12, 'HistoryComplete'))


class OracleAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(GITHUB_SERVER_URL='https://github.com',
                        GITHUB_REPOSITORY='sloik/superforecasting',
                        GITHUB_REF='refs/heads/main',
                        GITHUB_WORKFLOW_REF=oracle_job_guard.WORKFLOW,
                        GITHUB_EVENT_NAME='push')
        self.event = dict(repository=dict(id=1312500530, full_name='sloik/superforecasting', fork=False),
                          ref='refs/heads/main', deleted=False)

    def test_main_push_and_manual_dispatch(self):
        self.assertTrue(oracle_job_guard.admit(self.env, self.event))
        self.env['GITHUB_EVENT_NAME'] = 'workflow_dispatch'
        for ref in ('main', 'refs/heads/main'):
            self.event['ref'] = ref
            self.assertTrue(oracle_job_guard.admit(self.env, self.event))

    def test_foreign_context_is_denied(self):
        for key, value in [('GITHUB_REPOSITORY', 'foreign/repo'), ('GITHUB_REF', 'refs/heads/topic'),
                           ('GITHUB_SERVER_URL', 'https://example.test'),
                           ('GITHUB_WORKFLOW_REF', oracle_job_guard.WORKFLOW.replace('oracle-tla.yml', 'other.yml')),
                           ('GITHUB_EVENT_NAME', 'pull_request'), ('GITHUB_EVENT_NAME', 'pull_request_target')]:
            self.assertFalse(oracle_job_guard.admit(dict(self.env, **{key: value}), self.event))
        for field, value in [('id', True), ('id', 1312500530.0), ('id', 42),
                             ('full_name', 'foreign/repo'), ('fork', True)]:
            event = copy.deepcopy(self.event)
            event['repository'][field] = value
            self.assertFalse(oracle_job_guard.admit(self.env, event))
        for field, value in [('pull_request', {}), ('deleted', True), ('ref', 'refs/heads/topic')]:
            self.assertFalse(oracle_job_guard.admit(self.env, dict(self.event, **{field: value})))
        self.env['GITHUB_EVENT_NAME'] = 'workflow_dispatch'
        self.event['ref'] = 'topic'
        self.assertFalse(oracle_job_guard.admit(self.env, self.event))

    def test_duplicate_event_keys_fail_closed(self):
        with self.assertRaises(ValueError):
            json.loads('{"ref":"main","ref":"topic"}', object_pairs_hook=oracle_job_guard.unique_object)


class WorkerGuardTests(unittest.TestCase):
    def test_without_marker_is_remote(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(remote_formal.is_worker())

    def test_forged_or_wrong_platform_or_uid_or_cgroup_refused(self):
        marker = 'a' * 32
        for platform, user, group in [('darwin', 'tla-job', '/system.slice/tla-q-'+marker+'.service'),
                                      ('linux', 'root', '/system.slice/tla-q-'+marker+'.service'),
                                      ('linux', 'tla-job', '/system.slice/unrelated.service')]:
            with self.subTest(platform=platform, user=user, group=group), mock.patch.dict(os.environ, {'TLA_QUEUE_JOB':marker}, clear=True), mock.patch.object(remote_formal.sys, 'platform', platform), mock.patch.object(remote_formal.pwd, 'getpwuid', return_value=type('User', (), {'pw_name':user})()), mock.patch.object(Path, 'read_text', return_value=group):
                with self.assertRaises(RuntimeError):
                    remote_formal.is_worker()
        with mock.patch.dict(os.environ, {'TLA_QUEUE_JOB':'forged'}, clear=True):
            with self.assertRaises(RuntimeError):
                remote_formal.is_worker()

    def test_worker_requires_all_identity_facts(self):
        marker = 'a' * 32
        with mock.patch.dict(os.environ, {'TLA_QUEUE_JOB':marker}, clear=True), mock.patch.object(remote_formal.sys, 'platform', 'linux'), mock.patch.object(remote_formal.pwd, 'getpwuid', return_value=type('User', (), {'pw_name':'tla-job'})()), mock.patch.object(Path, 'read_text', return_value='0::/system.slice/tla-q-'+marker+'.service\n'):
            self.assertTrue(remote_formal.is_worker())


class QueueRoutingTests(unittest.TestCase):
    def test_missing_client_never_checks_local_jar_or_runs_java(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(remote_formal.shutil, 'which', return_value=None), mock.patch.object(run_formal.subprocess, 'run') as launch, mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('local JAR read forbidden')), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(78, run_formal.main(['--jar','/oracle/only/missing-locally.jar']))
        launch.assert_not_called()
        self.assertIn('no local TLC fallback', err.getvalue())

    def test_backend_failure_preserved_without_native_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(remote_formal.shutil, 'which', return_value='/client/tla-queue'), mock.patch.object(remote_formal.subprocess, 'run', return_value=subprocess.CompletedProcess([],75)) as launch:
            self.assertEqual(75, run_formal.main(['--jar','/oracle/only.jar','--java','/oracle/jdk/bin/java']))
        self.assertEqual(1, launch.call_count)
        command = launch.call_args.args[0]
        self.assertEqual('/client/tla-queue',command[0])
        self.assertEqual('sloik/superforecasting',command[command.index('--project')+1])
        self.assertEqual(['/usr/bin/python3','-B','tools/run_formal.py','--java','/oracle/jdk/bin/java','--jar','/oracle/only.jar'],command[command.index('--')+1:])

    def test_backend_os_error_is_explicit_no_fallback(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(remote_formal.shutil, 'which', return_value='/client/tla-queue'), mock.patch.object(remote_formal.subprocess, 'run', side_effect=OSError('backend unavailable')) as launch, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(75,run_formal.main(['--jar','/oracle/only.jar']))
        self.assertEqual(1,launch.call_count)
        self.assertIn('no local TLC fallback',err.getvalue())

    def test_whole_producer_snapshot_and_owned_output_merge(self):
        names = [case+'.log' for case,_ in run_formal.CASES]
        def backend(command, **kwargs):
            source = Path(command[command.index('--source')+1])
            expected = {'tools/run_formal.py','tools/remote_formal.py','formal/ForecastLifecycle.tla'}
            expected.update('formal/'+case+'.cfg' for case,_ in run_formal.CASES)
            self.assertEqual(expected,{str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()})
            self.assertEqual((ROOT/'tools/run_formal.py').read_bytes(),(source/'tools/run_formal.py').read_bytes())
            fetched=Path(command[command.index('--fetch')+1])/remote_formal.OUTPUT
            fetched.mkdir(parents=True)
            for name in names:
                (fetched/name).write_text('fresh '+name)
            (fetched/'unowned').write_text('must not copy')
            return subprocess.CompletedProcess(command,0)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'output';out.mkdir();(out/'unrelated.txt').write_text('keep')
            (out/names[0]).write_text('old owned log')
            with mock.patch.dict(os.environ,{},clear=True),mock.patch.object(remote_formal.shutil,'which',return_value='/client/tla-queue'),mock.patch.object(remote_formal.subprocess,'run',side_effect=backend):
                self.assertEqual(0,run_formal.main(['--jar','/oracle/only.jar','--output',str(out)]))
            self.assertEqual('keep',(out/'unrelated.txt').read_text())
            self.assertEqual('fresh '+names[0],(out/names[0]).read_text())
            self.assertFalse((out/'unowned').exists())
            self.assertEqual(set(names)|{'unrelated.txt'},{p.name for p in out.iterdir()})

    def test_snapshot_rejects_linked_producer_input_before_queue_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            for name in ('tools/run_formal.py','tools/remote_formal.py'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('safe fixture')
            (root/'formal').mkdir();(root/'formal'/'ForecastLifecycle.tla').symlink_to('/etc/passwd')
            args=Namespace(jar=Path('/oracle/only.jar'),java='/usr/bin/java',output=None)
            with mock.patch.object(remote_formal.shutil,'which',return_value='/client/tla-queue'),mock.patch.object(remote_formal.subprocess,'run') as launch,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(78,remote_formal.run(args,root,run_formal.CASES))
            launch.assert_not_called()

    def test_success_without_complete_fetched_campaign_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ,{},clear=True),mock.patch.object(remote_formal.shutil,'which',return_value='/client/tla-queue'),mock.patch.object(remote_formal.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(1,run_formal.main(['--jar','/oracle/only.jar','--output',str(Path(tmp)/'output')]))


class LogSafetyTests(unittest.TestCase):
    def test_destination_directory_and_owned_log_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'lifecycle.log').write_text('fresh')
            target=root/'target';target.mkdir();(target/'private').write_text('keep')
            symlink=root/'out';symlink.symlink_to(target,target_is_directory=True)
            with self.assertRaises(RuntimeError):remote_formal.merge_logs(source,symlink,['lifecycle.log'])
            (target/'lifecycle.log').symlink_to(target/'private')
            with self.assertRaises(RuntimeError):remote_formal.merge_logs(source,target,['lifecycle.log'])
            self.assertEqual('keep',(target/'private').read_text())

    def test_fetched_link_and_path_traversal_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'lifecycle.log').symlink_to('/etc/passwd')
            with self.assertRaises(RuntimeError):remote_formal.merge_logs(source,root/'out',['lifecycle.log'])
            with self.assertRaises(RuntimeError):remote_formal.merge_logs(source,root/'out',['../escape'])
            self.assertFalse((root/'escape').exists())

    def test_failed_campaign_can_retain_owned_partial_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'lifecycle.log').write_text('failure evidence')
            remote_formal.merge_logs(source,root/'out',['lifecycle.log','terminal.log'],require_complete=False)
            self.assertEqual('failure evidence',(root/'out'/'lifecycle.log').read_text())
            self.assertFalse((root/'out'/'terminal.log').exists())
