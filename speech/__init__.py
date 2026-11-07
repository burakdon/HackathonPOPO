"""Speech recognition for practice attempts: a small provider abstraction, a local Whisper provider,
and conservative transcript normalisation for *suggested* matching. Recognition output is only a
proposal; the helper confirms or corrects it before anything is scored."""
