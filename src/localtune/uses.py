"""What a model is used for, as the prompt and answer lengths a benchmark tests."""

# How fast a model feels depends on what it is asked to do, not only on the model and the
# computer. A chat question is short and gets a longer answer, so writing speed matters most;
# asking about code means pasting files, and asking about a document means pasting the whole
# document, so reading speed matters more and more. The settings that are fastest for one can
# lose for another (tune.py picks the settings that give a whole answer soonest), so "localtune
# run" and "localtune tune" can test a use with --use instead of a made-up length.
# --prompt-tokens and --gen-tokens still set the lengths by hand, and win over the use.

USES = {
    "chat": (512, 256),
    "code": (8000, 512),
    "documents": (16000, 256),
}
DEFAULT_LENGTHS = (512, 128)


class UnknownUseError(ValueError):
    pass


def token_lengths(
    use: str | None, prompt_tokens: int | None, gen_tokens: int | None
) -> tuple[int, int]:
    """The prompt and answer lengths to test: the ones given, else the use's, else the defaults."""
    if use is not None and use not in USES:
        raise UnknownUseError(f"Unknown use {use}. Choose from: {', '.join(USES)}")
    prompt, gen = USES[use] if use else DEFAULT_LENGTHS
    return (
        prompt if prompt_tokens is None else prompt_tokens,
        gen if gen_tokens is None else gen_tokens,
    )
