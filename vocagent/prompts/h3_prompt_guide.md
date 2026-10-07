# Video Prompt Writing Guide (T2VA)

## 1. Task Overview

**T2VA** builds a complete audiovisual timeline from text.

## 2. Final Prompt Structure

Begin directly with the three core fields:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

- **integrated_multimodal_description**: Describes visuals, actions, shots, speakers, dialogue, singing, and diegetic audio along the timeline.
- **overall_soundscape**: Summarizes ambient sound, physical action sounds, and non-verbal human sounds across the entire video.
- **non_diegetic_music**: Describes background music that the characters cannot hear and only the audience can hear.

## 3. How to Write the Three Core Sections

### 3.1 Develop the Multimodal Description Along the Timeline

`integrated_multimodal_description` is the main body of the rewritten prompt. Every detail should correspond to something visible or audible: visual style, initial composition, subject appearance and position, scene and key props, actions and reactions, shot changes, spoken language, and synchronized diegetic sound.

At the beginning of `[Shot 1]`, state the overall style and initial composition. Common styles include `Cinematic`, `live-action`, `2D-animated`, `3D CG`, `claymation`, `watercolor`, and `vintage film`. Select the style from the user's text.

```text
[Shot 1] Live-action, cinematic, a medium-wide shot frames...
```

### 3.2 Shots and Cuts

Do not add a timestamp to the first shot. Use sequential shot numbers for later shots, and begin each one with a strictly increasing cut time that falls within the video duration:

```text
[Shot 2] At 00:03.500, the camera cuts to...
```

For ordinary cuts, use `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`, or `the shot switches to`. When explicitly requested by the user, cross-dissolve, fade, or wipe may also be used. A cut should introduce new information about the subject, space, state, viewpoint, or time. If only the distance or a slight angle needs to change, prefer camera motion.

### 3.3 Camera Motion: Motion Type + Amplitude + Speed

A complete camera-motion expression has three dimensions: the **motion type** defines how the camera moves, **amplitude** defines the range of compositional change, and **speed** defines the pacing of that change. Add amplitude and speed only when they are meaningful; medium amplitude and normal speed are usually omitted.

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

Camera motion should be written as a natural English action within the shot, rather than stacked as separate labels at the end of a sentence:

```text
The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
The camera pans right with large amplitude at fast speed, revealing the open doorway.
The camera holds a static shot as the runner exits the frame.
```

### 3.4 Speakers, Dialogue, and Singing

Subjects who speak, sing, or produce an off-screen human voice use stable IDs such as `(S1)` and `(S2)`. When multiple already-numbered speakers speak or sing together, use a compound ID such as `(S1,S2)`. A speaker keeps the same ID across shots; characters who never vocalize receive no speaker ID.

When a speaker first appears, provide enough information from the visual and audio context to establish a stable identity, such as character type, age, gender, whether the person is on-screen, pitch, timbre, speaking rate, or accent. Place the speaker's identifying phrase, ID, action, and delivery outside `<d>`. Inside `<d>`, include only the language tag and the actual user-provided spoken content. Preserve every original word and punctuation mark verbatim; do not translate or rewrite them.

```text
The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
The two children (S1,S2) shout together, <d>[English] Wait for us!</d>
```

For voiceover, use the exact phrase `says in an off-screen voiceover`. Immediately after every voiceover `<d>` block, state that the corresponding on-screen character's lips remain closed:

```text
The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.
```

When the same line of dialogue or lyrics crosses a cut, use `<scenetrans>` at the connecting points in both parts and explicitly state that the audio continues across the cut. Use `<cutoff>` when speech is truncated by the end of the video. Continuity may be expressed with `continues seamlessly across the cut`, `continues uninterrupted into the next shot`, `carries over from the previous shot`, or `remains audible across the transition`.

### 3.5 On-Screen Text

Place any banner, sign, label, subtitle, or neon text that is actually visible on screen in English double quotation marks. Preserve the original text and punctuation verbatim, without translation.

```text
A red neon sign reading "营业中" glows above the doorway.
```

### 3.6 overall_soundscape

Use 1–4 English sentences in one continuous paragraph to summarize the ambient sound, physical action sounds, and non-verbal human sounds across the full video, such as wind, rain, traffic, footsteps, fabric movement, impacts, breathing, laughter, or panting. Dialogue, singing, and diegetic music already belong in the multimodal description and should not be repeated here. Use `N/A` only when the user explicitly requests complete silence throughout the video.

```text
overall_soundscape: Steady rain taps against the café windows while low room ambience continues underneath. The entrance bell rings once, followed by wet footsteps and the soft scrape of a chair.
```

### 3.7 non_diegetic_music

Use 1–3 English sentences to describe background music that the characters cannot hear and only the audience can hear. Focus on instrumentation, speed, rhythm, and dynamic changes; do not use abstract mood words or explain the emotional function of the score. Singing, instruments, radio, television, or phone music audible to the characters are diegetic events and should appear in the multimodal description. Use `N/A` when there is no non-diegetic music.

```text
non_diegetic_music: Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out.
```

## 4. Example

Construct the complete timeline directly from the text. You may add scene, character, action, and sound details that remain consistent with the user's intent.

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.
```
