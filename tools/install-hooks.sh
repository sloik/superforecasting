#!/bin/sh
# Install this repository's commit guard.
#
# Git never clones `.git/hooks`, so a fresh checkout has no protection until
# this runs. Run it once after cloning, before your first commit.
#
# Usage: tools/install-hooks.sh [--force]
#   --force   replace an existing hook that differs from the tracked source
#
# Idempotent: a second run with the guard already in place writes nothing.

set -u

USAGE="usage: tools/install-hooks.sh [--force]"

FORCE=0
while [ "$#" -gt 0 ]; do
	case "$1" in
	--force) FORCE=1 ;;
	-h | --help)
		echo "$USAGE"
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		echo "$USAGE" >&2
		exit 2
		;;
	esac
	shift
done

ROOT=$(git rev-parse --show-toplevel) || exit 1
cd "$ROOT" || exit 1

# Relative to the repo root, and correct inside linked worktrees.
HOOKDIR=$(git rev-parse --git-path hooks) || exit 1
SOURCE="tools/hooks/pre-commit"

if [ ! -f "$SOURCE" ]; then
	echo "missing hook source: $ROOT/$SOURCE" >&2
	exit 1
fi

mkdir -p "$HOOKDIR" || exit 1

FAILED=0
CHANGED=0

# One script under two names: git passes the message file to commit-msg only.
for name in pre-commit commit-msg; do
	target="$HOOKDIR/$name"

	if [ ! -e "$target" ]; then
		if cp "$SOURCE" "$target" && chmod +x "$target"; then
			echo "installed $name"
			CHANGED=1
		else
			echo "failed to install $name" >&2
			FAILED=1
		fi
	elif cmp -s "$SOURCE" "$target"; then
		echo "$name already up to date"
	elif [ "$FORCE" -eq 1 ]; then
		if cp "$SOURCE" "$target" && chmod +x "$target"; then
			echo "replaced $name (--force)"
			CHANGED=1
		else
			echo "failed to replace $name" >&2
			FAILED=1
		fi
	else
		echo "REFUSED: $name exists and differs from the tracked source." >&2
		echo "  tracked:   $ROOT/$SOURCE" >&2
		echo "  installed: $ROOT/$target" >&2
		echo "  diff them, then re-run with --force to replace it." >&2
		FAILED=1
	fi
done

if [ "$FAILED" -ne 0 ]; then
	exit 1
fi

if [ "$CHANGED" -eq 0 ]; then
	echo "nothing to do — the commit guard is already installed."
fi

exit 0
