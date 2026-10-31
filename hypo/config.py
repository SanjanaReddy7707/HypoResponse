"""Global switches.
SHOW_SNIPPET: whether the page includes the 3 most recent error/warning log lines.
Default False because the snippet leaks the root-cause category (a TF-IDF classifier reaches 100% from it vs ~40% from the
alert text alone), which would make tools/RAG/verifier nearly unnecessary. Use --show-snippet for the 'easy' setting."""
SHOW_SNIPPET = False
