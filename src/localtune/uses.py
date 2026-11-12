"""What a model is used for, as the prompt and answer lengths a benchmark tests."""

# How fast a model feels depends on what it is asked to do, not only on the model and the
# computer. A chat question is short and gets a longer answer, so writing speed matters most;
# asking about code means pasting files, and asking about a document means pasting the whole
# document, so reading speed matters more and more. The settings that are fastest for one can
# lose for another (tune.py picks the settings that give a whole answer soonest), so "localtune
# measure" and "localtune tune" can test a use with --use instead of a made-up length.
# --prompt-tokens and --gen-tokens still set the lengths by hand, and win over the use.

USES = {
    "chat": (512, 256),
    "code": (8000, 512),
    "documents": (16000, 256),
}
DEFAULT_LENGTHS = (512, 128)

# The exact lengths, written out for --help, so people know what each name tests.

USE_SIZES = "; ".join(
    f"{name} = {prompt:,} tokens in, {answer:,} out" for name, (prompt, answer) in USES.items()
)

# --use can also be a percent, like --use 30: a conversation (prompt and answer together) that
# fills that much of the biggest context the model can use on this computer. The same percent
# means a similar load for a small model and a big one, while the three named uses stay the same
# fixed lengths for every model. The answer is PERCENT_ANSWER tokens, or half the conversation
# when that is shorter, and the prompt is the rest.

PERCENT_ANSWER = 256


class UnknownUseError(ValueError):
    pass


def use_percent(use: str | None) -> int | None:
    """The percent in a use like "30" or "30%", or None for a named use."""
    if use is None or use in USES:
        return None
    number = use.removesuffix("%")
    if not number.isdigit() or not 1 <= int(number) <= 100:
        raise UnknownUseError(
            f"Unknown use {use}. Choose from: {', '.join(USES)}, or a percent of the "
            "context from 1 to 100, like 30."
        )
    return int(number)


def percent_lengths(percent: int, context: int) -> tuple[int, int]:
    """The prompt and answer lengths that fill this percent of a context."""
    total = max(context * percent // 100, 2)
    answer = min(PERCENT_ANSWER, total // 2)
    return total - answer, answer


def token_lengths(
    use: str | None,
    prompt_tokens: int | None,
    gen_tokens: int | None,
    context: int | None = None,
) -> tuple[int, int]:
    """The prompt and answer lengths to test: the ones given, else the use's, else the defaults.

    A percent use needs the model's usable context.
    """
    percent = use_percent(use)
    if percent is not None:
        if context is None:
            raise UnknownUseError(
                f"--use {use} needs to know the model's context, which only 'localtune "
                f"predict', 'localtune measure' and 'localtune tune' work out. Choose from: "
                f"{', '.join(USES)}."
            )
        prompt, gen = percent_lengths(percent, context)
    else:
        prompt, gen = USES[use] if use else DEFAULT_LENGTHS
    return (
        prompt if prompt_tokens is None else prompt_tokens,
        gen if gen_tokens is None else gen_tokens,
    )


def use_label(use: str | None) -> str | None:
    """How a use is saved with a run: a percent always ends with %, so "30" and "30%" match."""
    percent = use_percent(use)
    return f"{percent}%" if percent is not None else use
