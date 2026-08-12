#!/usr/bin/env bash
# Usage: bash check_quotes.sh your_commands.sh
# 1) flags curly/smart quotes and other non-ASCII that break shell parsing
# 2) checks each non-empty, non-comment line parses as a complete command
f="${1:?pass the script file}"

echo "=== non-ASCII / smart-quote scan (line:col:byte) ==="
# grep for anything outside printable ASCII; curly quotes are the usual culprit
grep -nP '[^\x00-\x7F]' "$f" && echo "^ FOUND non-ASCII chars (likely curly quotes) -- fix these" \
  || echo "none found (quotes are straight ASCII, good)"

echo
echo "=== per-command bash -n (syntax) check ==="
# each ablate_* invocation is one logical command; test them independently
n=0; bad=0
while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue          # skip blank
  [[ "$line" =~ ^[[:space:]]*# ]] && continue # skip comments
  n=$((n+1))
  if ! bash -n <<< "$line" 2>/tmp/e; then
    bad=$((bad+1))
    echo "LINE $n FAILS bash -n:"
    echo "  $line" | cut -c1-120
    echo "  err: $(cat /tmp/e)"
  fi
done < "$f"
echo "checked $n commands, $bad failed syntax"