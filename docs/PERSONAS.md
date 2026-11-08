# Demo patients: Maria and James

Two **composite personas** (not real people) used in the demo, the slides and the simulator. Their
numbers come from real data where it exists and are labelled as assumptions where it does not.
Simulator: `configs/sim_pnt.yaml`, presets `demo_maria` and `demo_james`. Check and agent results:
`uv run python -m scripts.demo_patients` -> [`results/demo_patients.md`](../results/demo_patients.md).

## Maria, 62: "the word is there, the sounds come out wrong"

- Eight months after a left-hemisphere stroke; retired teacher; lives with her husband.
- **Type:** moderate Broca's-type aphasia. Effortful speech; she knows the word but the sounds come
  out wrong ("cap" for "cat").
- **Naming ability:** Philadelphia Naming Test score about 49, the **40th percentile of 335 real
  people with aphasia** (data/pnt_thetas.csv).
- **Errors:** sound 61%, no response 24%, meaning 16% (simulated, from her two-step profile:
  meaning -> word strong, word -> sounds weak).
- **What helps:** the first-sound cue (+0.22 on tier-2 items), then first letter; a semantic cue
  barely helps (+0.02).
- **Practice:** tires after about 20 to 25 minutes; discouraged after three failures in a row.
- **Personal words (for the intake):** her grandchildren's names, garden plants, classroom objects.
- **In simulation (50 sessions):** fixed difficulty keeps her in the 75-90% band 38% of the time and
  she quits 12% of sessions; our agent 62% and 6%, with higher learning gain (3.75 vs 2.85 x100).

## James, 58: "the right word does not come, a related one does"

- Five months after a stroke; former bus driver; his daughter helps him practise.
- **Type:** semantic-type naming impairment. Sounds are intact; he picks a related wrong word
  ("dog" for "cat") or says nothing.
- **Naming ability:** PNT score about 50, the **44th percentile** of the same 335 real people.
- **Errors:** meaning 63%, no response 30%, sound 7%.
- **What helps:** a semantic cue (+0.16) or a sentence frame (+0.14); the first sound helps less (+0.08).
- **In simulation:** our agent beats fixed difficulty (band 67% vs 46%, learning gain 4.19 vs 3.40
  x100) **but after his meaning errors it still mostly gives the first-sound cue (54%)**, which is not
  what helps him most. **This is the demo moment for the clinician:** she overrides a few of those
  choices with a semantic cue, the agent learns from her (DITTO), and from then on it gives James the
  semantic cue after a meaning error.

## How each part is grounded

| part | source |
|---|---|
| Naming ability | Real PNT ability scores of 335 people with aphasia (MAPPD + R03; data/PNT_SOURCE.md) |
| Words and their difficulty | The 175 PNT words with difficulty measured on real patients (Fergadiotis et al. 2015) |
| Error profile and which cue helps | The two-step model of word retrieval (Dell et al. 1997; Foygel & Dell 2000) with these persona parameters (ASSUMPTION), reviewed by the team's clinical lead |
| How they speak (for the pitch) | Teaching videos: AphasiaBank Telerounds (public, cite MacWhinney et al. 2011) and Grand Rounds (watch only; do not download or redistribute) |
| Fatigue, frustration, life details | Assumptions / invented for the persona |

Say in the pitch: "Maria and James are composites. Their naming ability is placed on the scale of
335 real people with aphasia, and their error patterns follow the standard model of word retrieval."
