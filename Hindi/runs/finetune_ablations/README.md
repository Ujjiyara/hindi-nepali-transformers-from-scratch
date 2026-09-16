# Reasoning finetune: ablations that shaped the final configuration

These are the short runs made before the final finetune in `runs/finetune/`.
Each directory holds that run's `config.json`, `metrics.jsonl`, `train.log` and
the `finetune.yaml` it was run with. All start from the same pretrained
`best.pt` and are validated on held-out entity names.

| Run | Names in train pool | Embedding | Peak LR | Validation (step: answer loss / exact match) |
|---|---|---|---|---|
| Hindi `frozen_138names_300steps` | 138 | frozen | 3e-4 | 0: 6.185 / 0.0; 50: 1.128 / 0.393; 100: 1.174 / 0.365; 150: 1.045 / 0.35; 200: 0.989 / 0.377; 250: 0.892 / 0.361; 300: 0.857 / 0.424 |
| Hindi `frozen_48names` | 48 | frozen | 3e-4 | 0: 6.363 / 0.0; 50: 1.107 / 0.33; 100: 1.384 / 0.312; 150: 1.396 / 0.256; 200: 1.468 / 0.219; 250: 1.415 / 0.217; 300: 1.33 / 0.225 |
| Hindi `unfrozen_138names_300steps` | 138 | trained | 3e-4 | 0: 6.185 / 0.0; 50: 1.624 / 0.24; 100: 2.056 / 0.211; 150: 2.095 / 0.219 |
| Hindi `unfrozen_lr1e-4_48names` | 48 | trained | 1e-4 | 0: 6.363 / 0.0; 50: 1.278 / 0.234; 100: 1.565 / 0.195; 150: 1.765 / 0.207; 200: 1.908 / 0.213; 250: 1.994 / 0.184; 300: 1.944 / 0.205 |
| Hindi `v1_unfrozen_48names` | 48 | trained | 3e-4 | 0: 6.363 / 0.0; 125: 2.792 / 0.189; 250: 3.313 / 0.191 |
| Nepali `v1_unfrozen_48names` | 48 | trained | 3e-4 | 0: 7.342 / 0.0; 125: 2.828 / 0.18; 250: 3.941 / 0.158 |

**What they showed.**

1. `v1_unfrozen_48names` (both languages, stopped at step 250): validation
   answer loss rose from 2.8 to 3.3 (Hindi) and 3.9 (Nepali) while training
   loss fell. `v1_unfrozen_48names/diagnosis_step250.json` shows why: of 128
   held-out questions whose answer is an entity, 106 were answered with a
   *training* name that does not occur in the prompt. The finetune had turned
   the 48 training names into an answer vocabulary.
2. Freezing the tied embedding / output head removes the direct route for that
   shortcut (`frozen_*` against `unfrozen_*` at matched steps), and a larger
   name pool (the lexicons were enlarged from 80 to ~200 people per language)
   weakens it further. With both, 187 of 196 held-out entity answers named an
   entity from the prompt.
3. A lower learning rate alone did not help.
4. After 300 steps the model still chose among the prompt's entities at
   about chance, with training loss no lower than validation loss: it was
   underfitting the comparison, not overfitting. The final run therefore
   trains for 3,000 steps instead of 1,250.

The 48-name dataset these early runs used is reproducible from
`<Language>/configs/reasoning.yaml` at commit `e11c38d`; the `data` path in
their `config.json` refers to that version.
