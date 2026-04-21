#!/usr/bin/env sh
set -eu

usage() {
    echo "Usage: $0 <prefix>"
    echo "Example: $0 main_single"
}

if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

prefix=${1%.c}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
include_dir="$repo_dir/include"
lib_dir="$repo_dir/lib"
src="$script_dir/$prefix.c"
out="$script_dir/$prefix.out"

if [ ! -f "$src" ]; then
    echo "Error: source file not found: $src" >&2
    usage >&2
    exit 1
fi

mkdir -p "$lib_dir"

make -C "$include_dir" main
make -C "$include_dir" mpifor

${CC:-mpicc} ${CFLAGS:-} -Wall -O2 -std=c11 \
    -I"$include_dir" \
    -o "$out" "$src" "$include_dir/mpifor.o" \
    -L"$lib_dir" -lhhnet -lm

echo "Compiled: $out"
