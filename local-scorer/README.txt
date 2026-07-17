CAMS Scorer — local edition
============================

Score any country, company, or city on the eight CAMS nodes, using
your own OpenAI account. Nothing is sent to Neural Nations — this
runs entirely on your machine and talks directly to OpenAI.

SETUP (one-time)
-----------------
1. Get an OpenAI API key: https://platform.openai.com/api-keys
2. Set it as an environment variable before running:

   Windows (PowerShell):
       $env:OPENAI_API_KEY = "sk-..."

   Windows (cmd.exe):
       set OPENAI_API_KEY=sk-...

USAGE
-----
   cams-scorer.exe <country|company|city> <entity> <start_year> [end_year] [options]

Examples:
   cams-scorer.exe country USA 2020 2024
   cams-scorer.exe company Tesla 2023
   cams-scorer.exe city Detroit 2019 2023 --passes 5

Options:
   --passes N        number of isolated scoring passes (default 5)
   --min-passes N    minimum valid passes required per cell (default: min(3, passes))
   --model NAME      OpenAI model to use (default: gpt-5, or set OPENAI_MODEL)
   --out PREFIX      output filename prefix (default: derived from entity name)

OUTPUT
------
Two CSV files are written to the current folder:
   <entity>_cams_scores.csv     — ensemble mean per node (Coherence, Capacity, Stress, Abstraction)
   <entity>_cams_envelope.csv   — standard deviation + pass count per cell (uncertainty)

NOTES
-----
- Each pass is a separate live model call — a 5-pass run typically takes several minutes.
- This performs raw scoring only (no Node Value, Bond Strength, or diagnosis).
  See neuralnations.org for the full CAMS calculation pipeline and published datasets.
- Cost is billed to your own OpenAI account, at your account's usual rates.
