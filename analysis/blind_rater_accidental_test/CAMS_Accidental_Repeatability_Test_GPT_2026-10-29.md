# CAMS: An Accidental Test of Repeatability

*An accidental answer to spurious LLM anxiety about CAMS scores*

Research note, 29 October 2026
Kari McKern · ComplexityWorkz · JUNO/CAMS

## What I set out to do

I wanted to look at the American historical record through an unusual instrument.

CAMS describes a society as eight interacting organisational functions: Helm, Shield, Lore, Stewards, Flow, Craft, Hands and Archive. Each function is rated for Coherence (C), Capacity (K), Stress (S) and Abstraction (A). Together the ratings describe how a complex adaptive system is organised.

We already had an ensemble of AI-generated annual assessments for the United States, 1900 to 2026. The plan was to compare that series against 25 widely recognised periods of American history, from colonial America to the present. The periods mark shifts in the American zeitgeist: revolution, civil war, industrialisation, crisis, mobilisation, prosperity, disillusionment and institutional change.

The question was whether changes in collective social cognition can be recovered from the numerical behaviour of the eight functions. Could the instrument pick up a society becoming confident, frightened, divided, disoriented or newly able to act together? Could it show changes that the usual historical narratives had overlooked?

## How the test happened by accident

I didn't plan a repeatability test. GPT did, as a diversion from that study.

Partway through, GPT raised the familiar generic doubt that AI-generated scores may not be trustworthy observations. Instead of going straight to the comparison, it built a blinded test to probe that doubt. The GPT session built a blinded scoring exercise: 25 historical periods, eight functions each, 200 individual assessments. The casebook was anonymised and shuffled, and the raters never saw the original CAMS scores. The GPT session produced its own scored response, which I call the frozen GPT-session response. I then took the same blinded casebook to Claude (Opus, as recorded in RATER_DECLARATION.md) and to Grok, and each returned a full set of scores.

None of this was the intended experiment. It did put the doubt to a direct test. Can different AI raters, working independently on the same historical material, produce substantially similar assessments?

## The coordination expression

Each function was scored with:

Q = (A × C) + K − S

Abstraction multiplied by Coherence is the effective coordination of a function's symbolic and institutional capability. Capacity minus Stress is its operational surplus.

## Results

Claude's response was complete for 175 of the 200 node assessments, so all agreement figures below use the 175 node assessments that all three raters scored. That keeps every comparison on the same footing.

**Pairwise correlation of Q (Pearson r, n = 175)**

| Raters | r |
|---|---|
| Frozen GPT-session response and Claude | 0.931 |
| Frozen GPT-session response and Grok | 0.886 |
| Claude and Grok | 0.896 |

Frozen–Grok on all 200 assessments is 0.889.

**Absolute agreement.** The intraclass correlation is ICC(2,1), absolute agreement, single rater: 0.869 (95% bootstrap interval 0.833 to 0.897), on the 175 complete triples. This measures whether the raters give the same numbers, not only the same ups and downs.

**Why it sits below the consistency figure.** Consistency ICC (ICC(3,1)) is 0.899. The gap is calibration. Claude scores lower across the board: mean Q is 34.1, against 39.7 for the frozen GPT-session response and 40.5 for Grok. The raters agree on the shape of the pattern, and Claude sits a few points lower on the scale.

Averaged over the 25 periods, the three raters' period profiles correlate at 0.90 to 0.96.

## Do they recover the same changes between periods?

Yes, on the large movements. Mean Q change between selected periods, by rater:

| Transition | Frozen GPT-session | Grok | Claude |
|---|---|---|---|
| 1920s boom to Great Depression | −27.0 | −30.7 | −23.2 |
| Great Depression to New Deal | +22.8 | +27.2 | +19.0 |
| New Deal to WWII | +21.4 | +19.4 | +25.0 |
| 1991–2001 to 2020–25 | −19.4 | −23.3 | −26.0 |
| Civil War to Reconstruction | +3.3 | −7.2 | +0.9 |

The first four are the same story three times over. The fall into the Depression, the recovery through the New Deal, the surge of wartime mobilisation and the decline from the 1990s to now all appear in the same direction for every rater, at similar size.

Civil War to Reconstruction is a real disagreement. The frozen GPT-session response and Claude show a small rise. Grok shows a fall. Antebellum to Civil War splits the same way, with Grok rising slightly while the other two fall. These are the transitions where the raters read the same material differently, and they are worth a closer look in their own right.

The doubt GPT raised is answered. Three independent raters reproduce the same structure: the same relationships between functions and the same large movements between periods.

## A contemporary reading

The 2020–25 period gives a sharper result than I had claimed before. For all three raters, Craft is the highest function in the profile. Lore is close to the bottom of all 25 periods: it sits at the 8th, 4th and 8th percentile of Lore's own scores across periods for the frozen GPT-session response, Grok and Claude respectively. Helm is weak relative to the other functions in the same period, but it is not unusual against the long historical run (48th, 36th and 24th percentile). The finding that holds across raters is the gap between a very capable Craft and a depleted Lore.

## Returning to the original investigation

The repeatability result is a useful supporting finding. It doesn't replace the original question, which is about the relationship between measurable organisational states and historically recognised changes in collective consciousness.

The USA ensemble supplies the observations for 1900 to 2026. Earlier periods give historical context and room for separate reconstruction. The mapping of casebook cases to periods was taken from the cue text. The proposed periodisation is an external frame against which the ensemble can be tested.

The next analysis looks at the behaviour of the eight functions at historical turning points, using the operating JUNO graph-theoretic formalism. It should look at changes in the relationships between functions as well as the level of coordination.

- Does Helm become isolated before a crisis?
- Does Lore lose coherence while Archive and Craft keep accumulating capability?
- Does the system recover through renewed common purpose, institutional redesign or tighter coercive integration?
- Do similar configurations recur in different centuries?
- Can a change in organisational state be detected before contemporaries recognise that their circumstances have changed?

These are questions about the dynamics of social cognition, not about whether an AI can produce historical ratings.

## Conclusion

An unplanned exercise, set up by GPT in response to its own doubt about CAMS scores, answered that doubt. Three independent AI scoring responses to the same blinded casebook correlated at 0.89 to 0.93 on Q, with absolute agreement of 0.869 across the 175 fully scored node assessments. They recovered the same large historical movements, and they disagreed visibly where the history is contested.

The doubt is answered: three independent raters reproduce the structure. That is enough to proceed with the substantive work while keeping every individual rating open to challenge when the evidence calls for it.

The original ambition is larger. CAMS asks whether the changing configuration of a society's organisational functions is a quantitative record of its collective cognition, and whether the zeitgeist leaves a recognisable mathematical signature. The rating exercise supports the credibility of the instrument. The historical investigation will show what it can find.

## The standing CAMS methodological position

Going forward I will work to three principles.

1. **Ratings are taken as usable observations.** AI ratings are accepted as observations from a defined instrument. Their standing doesn't need to be argued again in every analysis. Generic reopening of the doubt that AI scores are valid observations is now treated as spurious, and it won't be raised again without specific evidence.
2. **Repeatable and open to challenge.** This exercise supports reproducibility. Specific results stay open to challenge, replication and revision.
3. **Discovery comes first.** The main task is to find organisational configurations, relationships, transitions, leading indicators and historical regularities, and to check downstream calculations against the operating JUNO graph-theoretic formalism.

I will separate findings from speculation, and I won't interrupt substantive research with generic qualifications about AI scoring.

The next step is the original experiment: compare the USA ensemble with the historical periodisation, looking in particular for shifts in the relationships between functions that could show changes in the American collective psyche.

The blind-rater result goes in the methodological record as support for the approach. It isn't the main discovery.
