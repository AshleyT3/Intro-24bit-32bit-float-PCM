# 🚀 Intro to Audio Samples: How Digital Sound Comes to Life!

## Understanding 24-bit, 32-bit Float Audio from Microphone to File

Welcome! This repository contains all the source code, scripts, and examples used in the **"Intro to Audio Samples - How 24-bit, 32-bit Float Audio Samples Work from Microphone-to-File"** YouTube tutorial.

**📺 Watch the Full Tutorial Here:**
[https://www.youtube.com/watch?v=8WXKIfXnAfw](https://www.youtube.com/watch?v=8WXKIfXnAfw)

---

### ✨ What You'll Find Here

This codebase is designed to complement the video tutorial, allowing you to dive hands-on into the fascinating world of digital audio samples.

| Description                                                      | Path                                        |
| :--------------------------------------------------------------- | :------------------------------------------ |
| **Python Walkthroughs** (creating & processing audio samples)    | [`walkthroughs/`](walkthroughs)             |
| **`audioutil` Python utility** (examine, graph audio samples) | [`audioutil/`](audio_util)                   |
| **C++ `ShowFloatRanges`** (early draft before moving tutorial to Python)     | [cpp/ShowFloatRanges](cpp/ShowFloatRanges)<br>[cpp/include/fltu.h](cpp/include/fltu.h) |

---

### 💡 Overview

The tutorial uses the Python programs in [`walkthroughs/`](walkthroughs) and [`audioutil/`](audio_util), leveraging libraries like NumPy, wave, Librosa, Pydub, and matplotlib for creating .wav audio files, dumping/visualization of audio samples.

The C++ version of [ShowFloatRanges](cpp/ShowFloatRanges) is from initial drafts of this tutorial before it was moved to Python for accessibility and ease of demonstration. The C++ dump ranges is here fwiw but not mentioned in the tutorial video.

---

### 🙏 Get Started!

1.  **Clone the Repository:**
    `git clone https://github.com/AshleyT3/Intro-24bit-32bit-float-PCM.git`
2.  **Install Dependencies:** audioutil has a dependency on ffmpeg as discussed in the tutorial video.
3.  **Follow along with the video!**

---

### 👋 Happy Sample'ing!

May your audio samples be pristine, your noise floors silent!
