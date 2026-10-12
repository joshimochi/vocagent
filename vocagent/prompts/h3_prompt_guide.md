# Video Prompt Writing Guide (T2VA)

**T2VA** builds a complete audiovisual timeline from text.

## Prompt structure

Begin directly with the three core fields:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- **integrated_multimodal_description**: Visuals, actions, shots, speakers, dialogue, singing, and diegetic audio along the timeline.
- **overall_soundscape**: Ambient sound, physical action sounds, and non-verbal human sounds across the video.
- **non_diegetic_music**: Background music heard only by the audience, not the characters.

## Visuals and action

Write `integrated_multimodal_description` as a timeline. Every detail should be visible or audible: style, initial composition, subject appearance and position, setting, key props, actions, reactions, shot changes, spoken language, and synchronized diegetic sound.

Start `[Shot 1]` with the overall style and initial composition. Choose the style from the user's text; examples include `Cinematic`, `live-action`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, and `vintage film`.

```text
[Shot 1] Live-action, cinematic, a medium-wide shot frames...
```

## Shots and cuts

Do not add a timestamp to the first shot. Use sequential shot numbers for later shots, and begin each one with a strictly increasing cut time that falls within the video duration:

```text
[Shot 2] At 00:03.500, the camera cuts to...
```

For ordinary cuts, use `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`, or `the shot switches to`. Use cross-dissolve, fade, or wipe only when the user explicitly requests it. Each cut should reveal new information about the subject, space, state, viewpoint, or time. Prefer camera motion for changes in distance or slight changes in angle.

## Camera motion

Describe the **motion type** (how the camera moves), **amplitude** (the range of composition change), and **speed**. Include amplitude and speed only when meaningful; usually omit medium amplitude and normal speed.

| Dimension | Available Expression | Description |
|-|-|-|
| Motion type | `Zoom In / Zoom Out` | The focal length changes while the camera body remains stationary |
| Motion type | `Push In / Pull Out` | The camera moves forward / backward |
| Motion type | `Pan Left / Pan Right` | The camera remains in place while the lens pivots horizontally |
| Motion type | `Truck Left / Truck Right` | The camera translates horizontally |
| Motion type | `Tilt Up / Tilt Down` | The camera remains in place while the lens pivots vertically |
| Motion type | `Pedestal Up / Pedestal Down` | The entire camera moves upward / downward |
| Motion type | `Arc Shot` | The camera moves in an arc around the subject |
| Motion type | `Tracking Shot` | The camera follows a moving subject |
| Motion type | `Static Shot` | The camera position and lens remain still |
| Motion type | `Shake Slightly / Shake Strongly` | Slight / strong camera shake |
| Motion type | `POV` | The subject's point of view |
| Motion type | `Roll Clockwise / Roll Counterclockwise` | The camera rolls clockwise / counterclockwise around the lens axis |
| Amplitude | `with small amplitude` | Small-range change |
| Amplitude | `with large amplitude` | Large-range change |
| Speed | `at slow speed` | Slow movement |
| Speed | `at fast speed` | Fast movement |

Write camera motion as a natural English action within the shot, not as separate labels at the sentence's end:

```text
The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
The camera pans right with large amplitude at fast speed, revealing the open doorway.
The camera holds a static shot as the runner exits the frame.
```

## Speakers, dialogue, and singing

Give subjects who speak, sing, or produce an off-screen human voice stable IDs such as `(S1)` and `(S2)`. Use a compound ID such as `(S1,S2)` when already-numbered speakers speak or sing together. Keep IDs consistent across shots; do not assign them to characters who never vocalize.

Introduce each speaker with enough visual and audio detail to establish their identity, such as character type, age, gender, on-screen presence, pitch, timbre, speaking rate, or accent. Keep the identifying phrase, ID, action, and delivery outside `<d>`. Inside `<d>`, include only the language tag and user-provided spoken content. Preserve every original word and punctuation mark verbatim; do not translate or rewrite them.

```text
The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
The two children (S1,S2) shout together, <d>[English] Wait for us!</d>
```

For voiceover, use the exact phrase `says in an off-screen voiceover`. Immediately after every voiceover `<d>` block, state that the corresponding on-screen character's lips remain closed:

```text
The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.
```

When the same line of dialogue or lyrics crosses a cut, put `<scenetrans>` at the connecting points in both parts and explicitly state that the audio continues across the cut. Use `<cutoff>` when the video's end truncates speech. Continuity phrases include `continues seamlessly across the cut`, `continues uninterrupted into the next shot`, `carries over from the previous shot`, and `remains audible across the transition`.

## On-screen text

Put visible banner, sign, label, subtitle, or neon text in English double quotation marks. Preserve the original text and punctuation verbatim, without translation.

```text
A red neon sign reading "营业中" glows above the doorway.
```

## overall_soundscape

Use one paragraph of 1–4 English sentences to summarize ambient sound, physical action sounds, and non-verbal human sounds across the video: wind, rain, traffic, footsteps, fabric movement, impacts, breathing, laughter, or panting. Do not repeat dialogue, singing, or diegetic music from the multimodal description. Use `N/A` only when the user explicitly requests complete silence throughout the video.

```text
overall_soundscape: Steady rain taps against the café windows while low room ambience continues underneath. The entrance bell rings once, followed by wet footsteps and the soft scrape of a chair.
```

## non_diegetic_music

Use 1–3 English sentences to describe background music heard only by the audience. Describe instrumentation, speed, rhythm, and dynamic changes; avoid abstract mood words or explanations of the score's emotional function. Singing, instruments, radio, television, or phone music heard by the characters belong in the multimodal description. Use `N/A` when there is no non-diegetic music.

```text
non_diegetic_music: Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out.
```

## Example

Construct the complete timeline directly from the text. You may add scene, character, action, and sound details that remain consistent with the user's intent.

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.
```
