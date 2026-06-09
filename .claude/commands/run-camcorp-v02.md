Run and verify the CAMS-CORP v0.2 pipeline.

Governing instructions: `docs/claude_code_camcorp_v02_instructions.md`.

Requirements:

1. Work in `C:\Users\julie\wintermute`.
2. Preserve all locked v0.2 constants in `src/camcorp5_pipeline/constants.py`.
3. Run the pipeline over:
   - `C:\Users\julie\Desktop\HariSeldon\cam5\camcorp5_ensemble_mean.csv`
   - `C:\Users\julie\Desktop\HariSeldon\cam5\camcorp5_envelope.csv`
4. Write outputs to `exports/camcorp5_pipeline/`.
5. Run the standard-library unittest smoke test.
6. Verify the anchor values listed in the instruction set.
7. If the test fixture is incomplete, update it to use all 8 CAMS nodes rather than weakening the v0.2 metric code.
8. Report commands run, files changed, and validation results.
