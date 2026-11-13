# Forecast lifecycle verification

Run the Python suite with `python3 -m unittest discover -s tests`.
Run the native model campaign with Java 17+ and the pinned
[TLC 1.7.4 release](https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4):

```sh
curl -fL https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar -o /tmp/tla2tools-1.7.4.jar
python3 tools/run_formal.py --jar /tmp/tla2tools-1.7.4.jar
```

The command checks SHA-256
`936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`
before running TLC, one worker, 512 MiB heap and a 120-second limit per case.
Use `--output <directory>` to retain fresh native logs with case, input hashes
and exit code. Logs are diagnostic receipts, not process attestations.

## Contract and bounds

`ForecastLifecycle.tla` is the shared contract for serialized **binary**
forecast commands. `lifecycle.cfg` explores two sequentially allocated IDs,
four accepted events, probabilities `{0, 0.5, 1}`, and outcomes `{0, 1}`.
The model uses integer probability units of one half and score units of one
quarter, avoiding floating point in the specification.
Rejected commands are stuttering steps: the Python correspondence test also
checks their nonzero return code and byte-for-byte unchanged temporary ledger.

| Invariant | Meaning | Semantic control |
| --- | --- | --- |
| `TypeOK` | All records have the declared types and bounds | Tool/type errors never qualify as a semantic counterexample |
| `TerminalImmutable` | Resolution/supersession seals the entire snapshot | `terminal.cfg` permits a terminal update |
| `HistoryComplete` | Every creation/update appears in the probability history; latest value agrees | `history.cfg` drops previous history |
| `LatestProbabilityScore` | Brier uses the final probability and recorded outcome | `score.cfg` uses the initial probability |
| `OnlyResolvedScored` | Open, absent and superseded forecasts have no score | `unscored.cfg` assigns superseded forecasts a score |
| `UniqueIds` | Each creation allocates a distinct ID | `ids.cfg` reuses ID 1 |

The safe model exhausts **2,029 states**. Each of the five negative controls
must exit with TLC's invariant-violation code and the exact intended invariant;
a parse failure, wrong invariant, extra error, or successful mutant fails the
campaign. No liveness or deadlock-freedom claim is made: reaching the configured
event limit is intentional, so deadlock checking is disabled.

`tests/test_forecastlog_formal.py` independently enumerates the same 2,029 trace
prefixes through the real Python command handlers, real temporary JSONL files
and real folding/scoring code. It checks accepted/rejected transitions, append
preservation, IDs, terminal state, complete history and Brier values. Existing
CLI and numeric tests remain in the normal suite.

This establishes bounded model safety and sampled numeric correspondence. It
is not an unbounded proof of Python correctness or forecasting accuracy.
Concurrent writers, crash durability, imported/legacy/malformed ledger rows,
ordered distributions and the other statistical algorithms are outside this
model. In particular, no locking or atomic multi-writer ID allocation is claimed.

## Oracle CI

`.github/workflows/oracle-tla.yml` runs tests and the fresh TLC campaign on the
repository-specific `oracle-superforecasting` ARM64 runner. Only maintained
`main` pushes and manual dispatches on `main` are eligible. Pull request tests
continue on GitHub-hosted machines.

The VM has a separate service account, registration and writable workspace for
this repo. Its service is `superforecasting-runner.service`; its application is
root-owned under `/opt/sf-runner`, with writes confined by systemd to
`/var/lib/sf-runner` and its diagnostic directory. It has no sudo access, one CPU
and a 2 GiB memory cap. Existing runner services are independent.

GitHub's 30-day update window is kept by the host: runner updates are applied by the host's verified updater (`swift-tla-runner-update.timer`, daily). GitHub's own auto-update stays disabled so the installation stays root-owned. The updater requires the release-notes SHA-256 and the asset digest to agree with the download, refuses a runner that no longer supports `ACTIONS_RUNNER_HOOK_JOB_STARTED`, never interrupts a running job, swaps only the runner code (registration, credentials, `.env`, hook and file modes are untouched), waits for the listener to reconnect, and restores the previous version on failure. After a real version change, run one real job before claiming compatibility. Tested on 2026-11-13 with a forced same-version swap followed by a green manual-dispatch run.

The root-owned pre-job admission hook at `/etc/sf-runner/oracle_job_guard.py`
is installed from `tools/oracle_job_guard.py`. It checks server, numeric repo
identity, repository name, main ref, exact workflow ref and push/manual event,
rejecting forks, PR payloads, deleted refs and duplicate JSON keys. Repository
settings require approval for all external fork contributors. This is a
maintained-main trust boundary, not a sandbox for hostile jobs: runner/action
preparation occurs before the pre-job hook, and administrators must review main.
See [GitHub's security guidance](https://docs.github.com/en/actions/reference/security/secure-use).

Runner **2.338.2** is installed from the upstream ARM64 archive pinned to
SHA-256 `670adb2f268fe21dcbb58b3db0e0aa9d5b3bddfe2c651481336e561ff86de945`.
Automatic runner updates are disabled because binaries are root-owned. An
administrator must install upstream updates within GitHub's supported update
window (normally 30 days, sooner for critical fixes), verify the archive and
restart this service. TLC is installed read-only at
`/opt/tla/tla2tools-1.7.4.jar`; the campaign checks its hash on every run.
Changes to the admission source need an explicit root-owned deployment; a
repository checkout cannot replace the installed hook. Verify deployed/source
hash equality and rerun admission allow/deny controls after any such deployment.

To disable this lane, an administrator can run:

```sh
sudo systemctl disable --now superforecasting-runner.service
```

Then remove this repository's runner registration in GitHub Settings → Actions
→ Runners and revert/disable only `oracle-tla.yml`. Leave the other services,
registrations and shared TLC artifact intact. No VM credentials belong in Git.
