#!/usr/bin/env bash
set -e

# prepend salus-server binary name if
# first arg is `-f` or `--some-option`
if [ "${1:0:1}" = '-' ]; then
	set -- salus-server "$@"
fi

if [ "$1" = 'salus-server' ]; then
    exec gosu salus "$@"
fi

exec "$@"
