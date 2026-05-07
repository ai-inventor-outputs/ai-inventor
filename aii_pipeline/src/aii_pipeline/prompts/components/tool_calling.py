"""Tool calling guidance components.

- get_tool_calling_guidance: Parallel tool call guidance for agents and LLMs
- get_web_tool_guidance: 3-level web tool hierarchy (WebSearch → WebFetch → aii_web_tools__fetch_grep)
"""


def get_tool_calling_guidance() -> str:
    """Returns guidance on efficient tool calling.

    Returns text wrapped in <tool_use> tags — callers should embed directly.
    """
    return """<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>"""


def get_web_tool_guidance() -> str:
    """Returns guidance on the 3-level web tool hierarchy.

    Returns text wrapped in <available_tools> tags — callers should embed directly.
    """
    return """<available_tools>
Three levels of web tools, broad to specific:

1. WebSearch — Returns titles, URLs, snippets. Use first to discover and scan the landscape.
2. WebFetch — A smaller LLM reads a page and returns a summary. HTML only, not PDFs. May miss specific details — use aii_web_tools__fetch_grep if it doesn't find what you need.
3. aii_web_tools__fetch_grep — Regex search over full document text (HTML/PDF). Returns exact matching sections with context. Use for precise details, exact numbers, methodology, or PDFs.

Workflow: WebSearch → WebFetch (understand) → aii_web_tools__fetch_grep (extract specifics).
</available_tools>"""
