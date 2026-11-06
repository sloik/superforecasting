"""Pre-job admission predicate, installed root-owned outside runner checkout."""

import json
import os
from pathlib import Path

REPOSITORY = "sloik/superforecasting"
REPOSITORY_ID = 1312500530
WORKFLOW = REPOSITORY + "/.github/workflows/oracle-tla.yml@refs/heads/main"


def admit(env, event):
    if env.get("GITHUB_SERVER_URL") != "https://github.com":
        return False
    if env.get("GITHUB_REPOSITORY") != REPOSITORY:
        return False
    if env.get("GITHUB_REF") != "refs/heads/main":
        return False
    if env.get("GITHUB_WORKFLOW_REF") != WORKFLOW:
        return False
    kind = env.get("GITHUB_EVENT_NAME")
    if kind not in {"push", "workflow_dispatch"} or "pull_request" in event:
        return False
    repo = event.get("repository", {})
    if type(repo.get("id")) is not int or repo["id"] != REPOSITORY_ID:
        return False
    if repo.get("full_name") != REPOSITORY or repo.get("fork") is not False:
        return False
    if event.get("deleted", False) is not False:
        return False
    if kind == "push" and event.get("ref") != "refs/heads/main":
        return False
    return not (
        kind == "workflow_dispatch"
        and event.get("ref") not in {"main", "refs/heads/main"}
    )


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate event key")
        result[key] = value
    return result


def main():
    try:
        event = json.loads(
            Path(os.environ["GITHUB_EVENT_PATH"]).read_text(),
            object_pairs_hook=unique_object,
        )
        accepted = isinstance(event, dict) and admit(os.environ, event)
    except (OSError, KeyError, ValueError, TypeError, AttributeError):
        accepted = False
    print(
        "Oracle maintained-main admission: " + ("accepted" if accepted else "rejected")
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
