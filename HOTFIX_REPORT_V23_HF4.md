# V23-HF4 PAC terminology update

- The inspection-only object is identified as **Patrol and Assessment Crew (PAC)**.
- The runtime class is `PatrolAssessmentCrew`, with the short alias `PAC`.
- The default object name is `PAC1`.
- The runtime collection is `pacs`.
- The configuration file is `PatrolAssessmentCrews_v1.json`.
- Graphical labels, text-interface examples, command schemas, tests, comments, error messages, and documentation use PAC terminology.
- MPS still inherits the same inspection and movement interface through `PatrolAssessmentCrew`; its existing behavior remains disabled or enabled exactly as before.
- No movement, inspection, analysis, repair, scoring, power-balance, command-validation, or game-ending logic was changed.
