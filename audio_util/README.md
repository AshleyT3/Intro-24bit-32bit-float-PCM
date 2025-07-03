# 🚀 audioutil - Analyze Audio Sample Ranges & Noise Floors

## Overview

`audioutil` is a standalone tool, also featured in the [Intro to Audio Samples](https://www.youtube.com/watch?v=8WXKIfXnAfw) tutorial. It offers the following capabilities:

* **Noise Floor Comparisons:** Analyze your own noise floor recordings to identify optimal equipment setups and observe differences across tests. Build a historical database of your results to compare equipment and arrangements over time. See `audioutil stats...` [walkthroughs](https://www.youtube.com/watch?v=8WXKIfXnAfw&t=26779s).
* **Stepped-Level Tests:** Record audio while a stepped-level test tone plays, then use `audioutil steadylevels...` to observe sample values at each level. Refer to the [test tone steady levels test](https://youtu.be/8WXKIfXnAfw?t=19745) in the tutorial.
* **Sample Range Dumps (24-bit & 32-bit float):** Understand 0.0 to 1.0 (0 dBFS) sample ranges and compare 24-bit with 32-bit floating-point ranges. See the `audioutil range simple` ([walkthrough](https://youtu.be/8WXKIfXnAfw?t=20854)) and `audioutil range detailed` ([walkthrough](https://youtu.be/8WXKIfXnAfw?t=22214)).
* **Plot Test Recordings:** Visualize `.wav` test recordings over time using Matplotlib. The straightforward implementation makes it a good starting point for graphing audio samples or learning Matplotlib.
* **Interactive IEEE-754 Mode:** Use `audioutil interactive` ([walkthrough](https://youtu.be/8WXKIfXnAfw?t=20313)) to input values and view detailed IEEE-754 interpretations.

---

⚠️ **IMPORTANT:** While a simple tool for the tutorial, `audioutil` proved invaluable for understanding environmental noise and how recorder settings/equipment arrangements impact noise floor tests. It's primarily designed for small `.wav` files (e.g., 5-10 second noise floor test recordings).

---

## Install

If you wish to install audioutil directly from PyPi (i.e., you are not following the tutorial), you can use the following command:

```pip install at-audioutil-pkg```

After installing audioutil, you can use `audioutil -h` to verify it is installed.

⚠️ Installing audioutil directly won't replace cloning the source code for the tutorial. To follow along with the tutorial video and walkthroughs, you'll need to clone the source code as instructed in the tutorial steps.

⚠️ FFmpeg Not Found? If you encounter an error that ffmpeg is not found, it means you likely don't have it installed. The tutorial [demonstrates](https://www.youtube.com/watch?v=8WXKIfXnAfw&t=18552s) resolving this for a Windows demo environment by using `winget install Gyan.FFmpeg`. 

**Important:** I don't endorse or vouch for the security of any FFmpeg builds, including the one shown. Please exercise caution when installing. For more information on available FFmpeg Windows builds, check the Windows download section at https://ffmpeg.org. As of July 2025, their site links to Gyan.FFmpeg builds, and I myself have had no issues with Gyan.FFmpeg to date.

---

## Tutorial

**📺 Watch the Full "Intro to Audio Samples" Tutorial Here:**
[https://www.youtube.com/watch?v=8WXKIfXnAfw](https://www.youtube.com/watch?v=8WXKIfXnAfw)

See the [main tutorial github page](https://github.com/AshleyT3/Intro-24bit-32bit-float-PCM) for additional details.

