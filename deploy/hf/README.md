---
title: NeuroCue
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Synthetic aphasia patient and RL therapy trainer demo
---

# NeuroCue

A clinically grounded synthetic aphasia patient and a reinforcement-learning therapy trainer,
built for the Triangle AI Hackathon (Health track).

The console walks through one patient: she names pictures with the error profile of a real
anomic speaker (semantic, phonological and nonword errors, false starts, carrier phrases), the
engine's fitter diagnoses her lesion from those answers, a trained policy designs her course, and
the last step shows how a clinician's correction enters the policy (DITTO).

The first request after a rebuild takes a minute or two while the engine loads.

Source, data provenance and results: https://github.com/burakdon/neurocue
