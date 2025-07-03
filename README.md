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

If you wish to install audioutil directly from PyPi, you can use the following command:

```pip install at-audioutil-pkg```

After installing audioutil, you can use `audioutil -h` to verify it is installed.

⚠️ Installing audioutil directly won't replace cloning the source code for the tutorial. To follow along with the tutorial video and walkthroughs, you'll need to clone the source code as instructed in the tutorial steps.

⚠️ FFmpeg Not Found? If you encounter an error that ffmpeg is not found, it means you likely don't have it installed. The tutorial [demonstrates](https://www.youtube.com/watch?v=8WXKIfXnAfw&t=18552s) resolving this for a Windows demo environment by using `winget install Gyan.FFmpeg`. 

**Important:** I don't endorse or vouch for the security of any FFmpeg builds, including the one shown. Please exercise caution when installing. For more information on available FFmpeg Windows builds, check the Windows download section at https://ffmpeg.org. As of July 2025, their site links to Gyan.FFmpeg builds, and I myself have had no issues with Gyan.FFmpeg to date.

---

### 👋 Happy Sample'ing!

May your audio samples be pristine, your noise floors silent!
