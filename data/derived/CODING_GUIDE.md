# Coding PSST naming responses: guide for the adjudicator

**What you are doing.** PSST recorded people with aphasia naming pictures (Boston Naming Test) and
actions (Verb Naming Test) and labelled each attempt only correct or incorrect. We need each
incorrect attempt sorted into the response types used in aphasia research, so we can check whether
our simulated patients make the same kinds of errors in the same proportions, and so the speech
scorer can learn to recognise them. A program has already coded everything and marked how confident
it is. **You only see the rows it was not sure about**, each with its guess and its reasoning.

**Time budget.** One to two hours. Do the BNT (noun) rows first; they are what the model is about.
If time runs out, stop; partial adjudication is useful.

**The file.** `data/derived/psst_adjudication.csv`. Fill `final_category` (leave blank to accept the
machine's `category`) and, when useful, `notes`. Columns `target_pron` and `transcript` are ARPAbet
phonemes with stress removed; `matched_word` is the dictionary word the transcript spells, if any.
Audio is in `data/raw/psst/<split>/<split>/audio/...` if you want to listen (filename is in the
original `utterances.tsv`).

## Categories

| code | meaning | example (target → response) |
|---|---|---|
| `correct_variant` | the target word in another form, or the target plus a suffix; PSST scored it strictly | comb → combs; stir → stirring |
| `semantic` | a real word related in meaning, not in sound | bench → chair; beaver → otter |
| `formal` | a real word related in sound, not in meaning (shares the first or last sound, a sound in the same position, or two sounds anywhere) | comb → coat; cactus → cactus-like "captain" |
| `mixed` | a real word related in both meaning and sound | octopus → otter; cat → rat |
| `unrelated` | a real word related in neither | house → table |
| `nonword_phon` | not a word, clearly an attempt at the target's sounds | toothbrush → "T UW B R AH SH" |
| `nonword_other` | not a word, shares nothing with the target | volcano → "M IH S AH" |
| `fragment` | a partial attempt that stops early | stethoscope → "S T EH" |
| `multiword` | a description, circumlocution, or several attempts strung together | unicorn → "horse with a horn"; send → "bailing the ... leder" |
| `no_response` | nothing said | |

## Reading the machine's columns

- `matched_word` with a leading `~` means a fuzzy match: the transcript is one phoneme away from
  that word ("CH EY R" → ~chair). Check it is the word the person meant.
- `reason` names the evidence: a WordNet similarity, a "shared hypernym", an "embedding cosine"
  (associative relatedness), the PNT formal criterion with the edit distance, or the length rule.

## Judgement calls the machine gets wrong

1. **Associative meaning.** WordNet knows taxonomy (beaver is an animal like otter) but not
   association (hammock → "sleep", tripod → "camera"). The machine marks these `unrelated` with a
   note to check. If a clinician would call it a semantic error, code `semantic`.
2. **Short nonwords that happen to spell a word.** "K AO T" for comb spells "caught". The machine
   calls it `formal`; if it reads as a sound attempt rather than a word choice, code `nonword_phon`.
   In the PNT convention a real-word output is a formal error regardless of intent, so lean `formal`.
3. **Segmentation.** The machine splits long transcripts into dictionary words greedily and can
   produce nonsense ("og to puss" for octopus). If the string is really one long attempt at the
   target, code `nonword_phon`; if it is a description, `multiword`.
4. **Verbs.** VNT responses are often whole phrases ("he's churning"). Code the head verb's relation
   to the target; a phrase that contains the target verb in another form is `correct_variant`.
5. **Formal relatedness on long targets.** The PNT rule counts two shared sounds anywhere as a
   formal relation. For long targets (unicorn, stethoscope, palette) almost any word qualifies, so
   the machine over-codes `formal` and `mixed`: "artist" for palette or "horse" for unicorn are
   really `semantic`. Ask whether a listener would hear the response as sounding like the target.
6. **When in doubt**, pick the category a speech-language pathologist would write on a PNT score
   sheet, and write a word in `notes`. Disagreement with the machine is the point.

## Round 2: the 50-row audit (about twenty minutes)

Your first 278 codes are in and counted (thank you). They showed the machine's `formal` and `mixed`
calls are unreliable, so a model trained on your codes (`scripts/learn_coder.py`) now codes the 111
noun rows you have not seen. Two things remain unchecked, and `data/derived/psst_audit_sample.csv`
is a 50-row sample chosen to check them:

- `audit_confident` (30 rows): errors the machine coded with high confidence that nobody has
  looked at. They are 20 phonological nonwords within three edits of the target and 10 WordNet
  semantic errors. Half of all real errors are rows like these, so this is the sample that decides
  whether the real nonword share we compare the engine against is right.
- `audit_learned` (20 rows): a random draw of the model's 111 codes, to measure its real accuracy
  (`learned_category` and `learned_prob` show its guess).

For this file, **fill `final_category` on every row**, even when you agree with the machine; a blank
here means not done. Same categories as before. Send the file back and it is merged by
`uv run python -m scripts.merge_adjudication data/derived/psst_audit_sample.csv`.

## What happens with your codes

They become the "real" column next to the engine's synthetic error mix by severity, and the
labelled set for the response-type classifier in the scorer. Your name goes on the method.
