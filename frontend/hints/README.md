# Hint audio

One folder per picture, five clips each, all generated with the macOS voice "Samantha"
(`scripts/make_hint_audio.py`, decision D7):

```
frontend/hints/<exercise_id>/first_sound.m4a      "It starts with the sound ..."
                             first_letter.m4a     "It starts with the letter ..."
                             sentence_frame.m4a   "A pet that says meow is a blank."
                             semantic.m4a         the meaning hint
                             whole_word.m4a       the word itself (shown after the last try)
```

The app finds a clip by this path (`GET /pictures` returns `audio_url` for every clip that exists), so
nothing else changes when a clip is replaced.

## Replace a clip

Put a recording with the same name in the same folder: `frontend/hints/ex-cat/first_sound.m4a`. Any
length is fine; AAC in an `.m4a` container plays everywhere. To convert another format:

```
afconvert -f m4af -d aac recording.wav frontend/hints/ex-cat/first_sound.m4a
```

Re-running the generator **skips clips that already exist**, so a recording you dropped in survives.
`--force` regenerates (and overwrites) every clip:

```
uv run --extra llm --extra speech python -m scripts.make_hint_audio            # only the missing ones
uv run --extra llm --extra speech python -m scripts.make_hint_audio --force    # everything again
```

## First sounds should be re-recorded by a person

Text-to-speech says isolated sounds badly: "k" comes out as the letter name, and "tr" or "gl" as
a clumsy word. The `first_sound` clips are the best candidates for a human recording ("kuh",
"tr", "gl", spoken the way a therapist would). The other clips are ordinary sentences and words
and are fine as generated.

Hackathon only: if the clips are ever shared outside the team, check the license for macOS voices.
