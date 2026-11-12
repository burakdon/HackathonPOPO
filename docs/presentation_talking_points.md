# NeuroCue: what to say on stage

A checklist for the Triangle AI Hackathon (Health track) presentation. Every number below comes from
a results file in this repository (named in brackets). Say the pitch, walk the pipeline in order,
run the five demo steps, close on the physician loop and the limits. Twelve minutes fits.

Live demo: https://burakdon.github.io/neurocue/ (six precomputed patients; loads instantly).
Repository: https://github.com/burakdon/neurocue. Video: results/demo/neurocue_demo.mp4.

## The pitch (say it first, in one breath)

NeuroCue is a clinically grounded synthetic aphasia patient you can run therapy experiments on, a
therapist policy trained against it, and a loop that lets the clinician correct that policy. Every
layer is fitted to published data and checked on held-out data; everything on screen is the same
code the validation ran.

## The problem (two sentences)

Anomia therapy is a sequence of decisions: which word next, which cue, retry or move on, how to
space practice. Reinforcement learning could optimise that, but it needs a patient to practise on,
and no simulator existed that was grounded in real naming errors and real retention data.

## The pipeline: say each layer, what it is fitted to, and how it was checked

1. **Data.** PSST (Gale et al. 2022; TalkBank/AphasiaBank): 3,291 naming utterances from 107
   sessions of about 75 people with aphasia, Boston Naming Test and Verb Naming Test, with ARPAbet
   transcripts and WAB-R aphasia quotients. PSST labels correctness only, so error *types* had to be
   built. 1,074 BNT noun rows drive the model; 74 training participants fit it, 33 held-out
   participants check it. [data/README.md, results/engine_validation.md]

2. **Manual data labeling (must say: this is the human-in-the-loop part of the data).** A clinician
   teammate adjudicated the rows the machine coder was unsure about: 278 rows in a first pass plus
   a 50-row audit, 328 clinician-coded responses in all. Her shorthand on the sheets: "y" means the
   production was effectively correct, "xxx" means she said a different word that is itself a test
   item. The audit covered 30 high-confidence machine codes (all 10 semantic calls held; 17 of 20
   phonological nonwords held, 3 were productions she judged correct that PSST had scored wrong) and
   20 of the learned coder's own calls (8 right). On the uncertain rows the rules agreed with her
   only 32% of the time, and only 3% of the machine's "formal" codes survived, so the labels changed
   the model. Precedence downstream: clinician > learned coder > rules. Say: her name goes on the
   method. [results/psst_coder_agreement.md, patientmodel/README.md]

3. **Learned response coder.** A random forest on 42 features of each target/response pair
   (normalised edit distance, MiniLM embedding cosine, WordNet similarity, length difference,
   shared-position ratio, the machine's own code) trained on the 328 clinician rows, cross-validated
   leaving one target picture out: 51% on nine categories and 68% on the five semantic-phonological
   error types, against 35% and 42% for the rules. It then recoded the 91 remaining uncertain rows.
   [results/psst_coder_learned.md]

4. **The patient: Dell's semantic-phonological model.** The interactive two-step model of word
   retrieval (Dell, Schwartz, Martin, Saffran and Gagnon 1997; Foygel and Dell 2000; Dell et al.
   2004, 2013) with two lesion weights, s (meaning to word) and p (word to sounds). Our
   implementation reproduces Dell's published patient error proportions to rmsd .002 to .011 and
   refitting his 14 patients recovers their s and p to .002. A patient's 4,000 trials simulate in
   about 80 ms. [patientmodel/README.md]

5. **Population prior fitted to real patients.** Dell's Philadelphia prior (log p/s mean +0.5,
   sd 0.6) did not match PSST, so we fitted the prior by multinomial likelihood on the training
   participants' error types: log p/s mean -0.2, sd 0.3 (deviance 164.8 against 279.5 for Dell's
   values; his prior is kept as a named profile). On the 33 held-out participants the engine's
   nonword share is 0.56 against a real 0.53, where Dell's prior gives 0.33; naming accuracy by
   severity band tracks the real curve (0.30/0.44/0.58/0.66 synthetic against 0.30/0.35/0.45/0.65
   real, with binomial noise of about 0.12). Say the limit out loud: the model over-produces formal
   and unrelated errors and under-produces semantic ones because the SP model's six-word
   neighbourhood has two formal and two unrelated competitors against one semantic; documented, not
   tuned away. [results/population_prior_fit.md, results/engine_validation.md]

6. **Words, pictures and how she talks.** IPNP picture norms (Bates et al. 2003; Szekely et al.
   2004: 520 objects, 244 drawings), Brysbaert et al. 2014 concreteness (about 4,800 concrete
   nouns), CMUdict pronunciations, WordNet (noun filter, semantic relations), wordfreq (Speer 2022),
   all-MiniLM-L6-v2 embeddings. Carrier phrases and fillers were mined from the PSST transcripts by
   severity, so the avatar says things like "okay that's a... home" the way real speakers do.
   [patientmodel/README.md, data/calibration/psst_utterance_frames.json]

7. **Learning layer: ACT-R fitted to a real therapy study.** Pavlik and Anderson's (2005)
   activation model with per-trace decay that depends on activation at practice, which produces the
   spacing effect mechanically rather than by rule. Fitted to Middleton, Rawson and Verkuilen 2019
   (four people with aphasia, 192 items each, retrieval practice versus repetition, spaced versus
   massed, tests at 7 and 28 days; 56 observed cells). Fitted trace weights: unaided success 1.00,
   failure followed by the word 0.59, study 0.58. The worst pooled cell is off by 0.08 and the fit
   reproduces the spacing effect, the retrieval-practice advantage, the rise across sessions and
   the forgetting between one week and one month. Caveat: four anomic patients and proper nouns.
   [results/learning_fit.md]

8. **Cues as activation.** First sound, last sound, meaning cue and whole word enter the SP model
   as external activation. Levels were calibrated to Nardo et al. 2017 rescue shares: first sound
   0.70 (target 0.70), whole word 0.82 (target 0.825). The final-sound cue cannot reach its target
   (0.24 against 0.475): a documented model limit, because the formal competitors share the coda.
   Untuned predictions that match the literature: cue responsiveness falls with severity (Pease and
   Goodglass 1978 found phonemic cues rescued 49% of failed pictures, less in severe patients). One
   that contradicts it: semantic cues help semantic lesions most, the opposite of Python et al.
   2021, so we flag that mechanism as suspect. A cued success lays a weaker memory trace than an
   unaided one (direction from Howard et al. 1985), an assumption, not a fit.
   [results/cue_calibration.md]

9. **Therapy environment and trainer.** A Gymnasium environment on the engine: one episode is a
   course of four sessions on days 0, 3, 7 and 10, 40 trials each, on 20 personally hard words;
   16 actions (which word: due, repeat, weakest, strongest; which cue: none, first sound, meaning,
   whole word) with hard action masks; the observation is the recent history, so it is a POMDP. The
   reward is the change in predicted retention after each trial, which telescopes exactly to the
   objective: what is still named 7 and 28 days after the course (Middleton's design). Trained with
   MaskablePPO (Stable-Baselines3 and sb3-contrib; Schulman et al. 2017; Huang and Ontanon 2022;
   Raffin et al. 2021), 160k steps, minutes on a laptop CPU. [results/engine_trainer.md]

10. **Results against clinical baselines** (80 held-out test patients, retention gain at one week):
    trained policy +0.315, spaced retrieval +0.315, the clinic's cueing hierarchy +0.293, errorless
    +0.256, random +0.247, massed +0.228, weakest-first +0.218. The policy beats the hierarchy by
    +0.022 (95% interval +0.020 to +0.026, paired) and cues on 1% of trials: nobody told it about
    spacing or retrieval practice; it rediscovered spaced, uncued retrieval practice, the best
    clinical practice, and found nothing better. Say the honest framing: the learning layer was
    fitted to retrieval-practice data, so spaced beating massed is built in; what is not built in is
    how a policy trades a cue-assisted success now against a weaker trace later. This is the
    engine's current opinion, not a verdict on cueing. [results/engine_trainer.md]

11. **DITTO: where the physician comes in.** Shaikh et al. (ICLR 2025): a clinician's correction is
    a demonstration, preferred over every action the policy samples in that state, and the policy is
    updated with the DPO loss (Rafailov et al. 2023) against a frozen copy of itself. No LoRA is
    needed: the actor is a small network. Our addition is anchors, a penalty on states the clinician
    did not mean. Plain DITTO over-generalised (a first-sound cue after every error, leakage 0.38 to
    1.00); anchored DITTO adopted the rule on 96% of matching held-out states and left 99% of other
    decisions unchanged; a fresh live validation switched the top action on 89 to 98% of matching
    states. On the NeuroCue policy one demonstration teaches nothing beyond its own state (0%
    adoption), four to six consistent ones teach the rule (76 to 100% adoption). Frame it as
    infrastructure: because the policy is already optimal under the fitted learning layer, a
    correction cannot raise predicted retention, so the demo judges a correction on two things only,
    adoption where it applies and containment everywhere else. That is customisability with full
    transparency. [README.md DITTO section, results/live_ditto_validation.md, neurocue/demo/server.py]

12. **The demo itself.** FastAPI over the engine, a hand-drawn SVG avatar (Marisol, 23) whose mouth
    follows the phonemes she produced, browser speech synthesis, hosted on Modal from the same
    Dockerfile. The page says it and you should too: everything on screen is the validated engine,
    live, not a recording.

## Demo actions (click, then say)

1. **Who are you listening to.** Set AQ 55, phonological, seed 3, New patient. Say: two lesion
   weights drawn from the prior fitted to PSST.
2. **She names pictures.** Click Name it five or six times with sound on; let one nonword and one
   false start play. Say: the avatar mouths the phonemes she actually produced. Then type 14 into
   the batch box and press Go. Say: or skip the listening.
3. **Diagnose.** Point at the posterior heat map. Say: from 20 answers alone. Click Reveal true
   lesion. Say: the truth sits inside the bright region; the same fitter reproduced Dell's patients.
   Then AQ 50, semantic, seed 3, New patient, name a few: the errors become word substitutions and
   the posterior moves to the semantic side.
4. **Two therapists.** Run a four-session course. Say: identical copies of her, the clinic's cueing
   hierarchy against the trained policy, judged on what is still named a month later. Read the
   result line. Point at the strip: which word, which cue, trial by trial; the policy learned
   spacing on its own.
5. **Where the physician comes in.** Play one correction (the loop animation). Move the two sliders
   (illustration, say so). Then teach a rule live: after a nonword, retry the word with its first
   sound; press Teach it. Read the two chips: adopted where meant, contained elsewhere. Say: the
   policy took the clinician's rule and nothing more.

## Ten numbers to know cold

| What | Number | Source |
|---|---|---|
| Real utterances / participants | 3,291 / about 75 (107 sessions) | data/README.md |
| Clinician-labelled responses | 328 (278 + 50 audit) | results/psst_coder_agreement.md |
| Rules vs clinician agreement on uncertain rows | 32% | results/psst_coder_agreement.md |
| Learned coder, five SP error types (cross-validated) | 68% vs 42% rules | results/psst_coder_learned.md |
| Fitted prior | log p/s mean -0.2, sd 0.3 | results/population_prior_fit.md |
| Held-out nonword share, engine vs real | 0.56 vs 0.53 (Dell's prior 0.33) | results/engine_validation.md |
| Learning-layer trace weights | success 1.00, fail+word 0.59, study 0.58 | results/learning_fit.md |
| Cue calibration | first sound 0.70, whole word 0.82, final sound 0.24 (limit) | results/cue_calibration.md |
| Retention gain at one week | policy +0.315, hierarchy +0.293, massed +0.228 | results/engine_trainer.md |
| Anchored DITTO | 96% adoption, 99% of other decisions unchanged | README.md DITTO section |

## Literature and tools we built on (one line each)

- Dell, Schwartz, Martin, Saffran and Gagnon 1997; Foygel and Dell 2000; Dell et al. 2004, 2013: the
  two-step and semantic-phonological models, their fitted patients, dual-route repetition.
- Schwartz, Dell, Martin, Gahl and Sobel 2006: case-series test of the two-step model.
- Walker 2016 (WebFit): Bayesian fitting of the SP model.
- Gale et al. 2022: the PSST corpus (TalkBank, AphasiaBank).
- Middleton, Rawson and Verkuilen 2019; Middleton et al. 2016: retrieval practice and spacing in
  anomia therapy, the learning-layer fit target.
- Pavlik and Anderson 2005: ACT-R activation with activation-dependent decay.
- Nardo et al. 2017: cued-naming rescue shares, the cue calibration target.
- Pease and Goodglass 1978; Howard et al. 1985; Python et al. 2021; Meteyard and Bose 2018: cue
  effects by severity, by lesion type, and over time.
- Nickels and Howard 1995; Nickels 2002: frequency, length and item-specific therapy gains.
- Bates et al. 2003; Szekely et al. 2004: IPNP picture norms. Brysbaert, Warriner and Kuperman 2014:
  concreteness. Speer 2022: wordfreq.
- Schulman et al. 2017 (PPO); Huang and Ontanon 2022 (invalid-action masking); Raffin et al. 2021
  (Stable-Baselines3, sb3-contrib); Gymnasium.
- Rafailov et al. 2023 (DPO); Shaikh et al., ICLR 2025 (DITTO).
- scikit-learn, CMUdict, WordNet via NLTK, sentence-transformers (all-MiniLM-L6-v2), FastAPI, uv,
  Modal.

## Limits to say before anyone asks

- A prototype, not a clinical tool: no claim of effectiveness, diagnosis or assessment of impaired
  speech. No real patient is simulated; the patient is a model fitted to population data.
- The error-type split (semantic vs formal vs unrelated) is wrong in a known way, caused by the SP
  model's neighbourhood; the nonword share and severity curve are right.
- The learning layer rests on four anomic patients and proper nouns; cue-effort weights are
  assumptions; the semantic-cue mechanism is suspect.
- The trainer's advantage over the cueing hierarchy is real but small (+0.022), and a correction
  cannot beat an already optimal policy on this engine, which is why DITTO is shown as adoption and
  containment, not as a score.
- One PPO seed. Clinician labels come from one adjudicator.

## Do not say

- "Validated therapy" or "clinically proven". Say "fitted to published data and checked on held-out
  participants".
- Do not mix the older session-level simulator results (85% band, cohorts, abandoned sessions) with
  the NeuroCue course results; they are different environments.
- Do not claim DITTO improved retention; it did not, by design of the engine.
- Do not call the avatar's speech real patient audio; it is the model's phonemes rendered by the
  browser.

## Likely questions

- *Why not train on real patients?* No consented, item-level therapy logs exist at this scale; the
  engine is the bridge, and every layer names the real data it is fitted to.
- *How do you know the simulator is right?* Held-out PSST participants (nonword share, severity
  curve), Dell's published patients (rmsd .002 to .011), Middleton's retention cells (within 0.08),
  Nardo's cue effects (two of three reached). The failures are listed, not hidden.
- *What did the policy learn?* Spaced, uncued retrieval practice, which is the best clinical practice
  in the literature it was never shown; it cues on 1% of trials.
- *What is DITTO for if it cannot improve the score?* Turning a clinician's judgement into policy
  behaviour with a guarantee of containment; the moment the engine's cue assumptions are replaced by
  cued-therapy retention data, the same loop prices the corrections.
- *What would you do next?* Cued-therapy retention data to fit the cue-effort weights; a richer
  phonological neighbourhood to fix the error split; more labelled responses with a second
  adjudicator; multiple PPO seeds.
