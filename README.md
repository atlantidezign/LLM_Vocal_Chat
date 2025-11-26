# 🎙️ LLM_Vocal_Chat: Voice Chat with LM Studio

A complete Python application with graphical interface to interact vocally with language models via LM Studio.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Key Features

### 🎨 Modern Graphical Interface
- **Gradio Web UI**: Clean and intuitive browser-based interface
- **Real-time chat**: Messenger-style conversation view
- **Responsive layout**: Organized in tabs for ease of use
- **Multi-language UI**: Auto-detects browser language (English/Italian)

### 🎮 GPU Acceleration (NVIDIA)
- **Whisper on GPU**: 10-20x faster with CUDA
- **FP16 support**: Optimal performance on RTX series
- **Real-time GPU info**: Displays detected GPU and status
- **Automatic recommendations**: Suggests optimal models for your GPU

### 🎤 Multiple Input Methods
- **Voice recording**: Record directly from microphone via interface
- **Text input**: Write your messages as an alternative
- **Dual voice recognition**:
  - **Google Speech Recognition**: Online, fast (default)
  - **Whisper by OpenAI**: Local, more accurate, 99+ languages, complete privacy ⭐
- **Language selection**: Support for Italian, English, Spanish, French, German and many more

### 🤖 Advanced LM Studio Integration
- **Model selection**: Detects and allows choosing between loaded models in LM Studio
- **Connection test**: Verify server status with one click
- **Parameter configuration**: Temperature, max tokens, customizable system prompt
- **Full API support**: Compatible with LM Studio's OpenAI API

### 🔊 Customizable Voice Synthesis
- **Voice selection**: Choose from all available system voices
- **Speed control**: Slider from 50 to 300 WPM
- **Volume control**: Adjust from 0 to 100%
- **Audio interruption**: Button to stop overly long responses

### 📊 Statistics and Monitoring
- **Response time**: Measurement for each message
- **Message counter**: Tracks number of interactions
- **Average time**: Automatic performance calculation
- **Real-time status**: Visual indicators of application state

### 💾 Conversation Management
- **JSON saving**: Export conversations in structured format
- **TXT export**: Readable format for quick reference
- **AI Summary Document**: 🆕 Automatically generates a structured markdown document with:
  - Summary of main topics
  - Key points and decisions
  - Action items and next steps
  - Relevant technical details
- **Automatic timestamps**: Files named with date and time
- **Conversation reset**: Clear chat with one click
- **Markdown preview**: View summary before download
- **Persistent memory**: 🆕 Automatically saves and restores sessions
- **DOCX export**: 🆕 Professional Word documents

### 🌍 Multilingual
- English 🇬🇧 (default)
- Italiano 🇮🇹

## 📋 Requirements

- **Python** 3.7 or higher
- **LM Studio** installed and configured
- **Working microphone**
- **Internet connection** (only for Google voice recognition and first Whisper model download)
- **Modern web browser** (Chrome, Firefox, Edge, Safari)
- **NVIDIA GPU** (optional but highly recommended for Whisper)

## 🚀 Installation

### 1. Clone or download the project

```bash
git clone <repository-url>
cd LLM_Vocal_Chat
```

Or download the files:
- `vocal_chat.py` - Main script
- `requirements.txt` - Python dependencies
- `README.md` - This guide

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
or
```bash
# Base + scipy
pip install gradio SpeechRecognition pyttsx3 requests numpy scipy
pip install pipwin
pipwin install pyaudio
pip install python-docx

# With Whisper and GPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/whisper.git

# Optional (improves resampling)
pip install librosa
```

Complete commands:
```bash
# 1. Install ffmpeg
choco install ffmpeg

# 2. Install everything with requirements
pip install -r requirements.txt

# 3. PyAudio on Windows
pip install pipwin
pipwin install pyaudio

# 4. Whisper
pip install git+https://github.com/openai/whisper.git

# 5. Launch!
python vocal_chat.py
```

#### 🪟 Problems with PyAudio on Windows?

If you get errors during `pyaudio` installation on Windows:

**Method 1 - Pipwin (Recommended):**
```bash
pip install pipwin
pipwin install pyaudio
```

**Method 2 - Precompiled wheel:**
1. Download the appropriate .whl file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
2. Install with: `pip install PyAudio‑0.2.11‑cpXX‑cpXX‑win_amd64.whl`

#### 🍎 macOS - Complete Installation:

```bash
# 1. Install ffmpeg
brew install ffmpeg

# 2. Install PortAudio (for PyAudio)
brew install portaudio

# 3. Install Python libraries
pip install gradio SpeechRecognition pyttsx3 pyaudio requests numpy scipy python-docx

# 4. PyTorch (with GPU support if you have Apple Silicon Mac)
pip3 install torch torchvision torchaudio

# 5. Whisper
pip install git+https://github.com/openai/whisper.git
```

#### 🐧 Linux - Complete Installation:

```bash
# 1. Install ffmpeg and PortAudio
sudo apt update
sudo apt install ffmpeg portaudio19-dev python3-pyaudio

# 2. Install Python libraries
pip install gradio SpeechRecognition pyttsx3 pyaudio requests numpy scipy python-docx

# 3. PyTorch with CUDA (for NVIDIA GPU)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Whisper
pip install git+https://github.com/openai/whisper.git
```

#### 🐍 Python Environment Setup (optional)
Follow these steps to prepare your environment:

1. **Install Anaconda**: Follow Anaconda installation instructions for your OS, available on the [Anaconda website](https://www.anaconda.com/).
2. **Create a new Conda environment**:
```bash
conda create -n vocal_chat313 python=3.13.0
```

Replace `vocal_chat313` with a name of your choice.

3. **Activate the environment**:
```bash
conda activate vocal_chat313
```

#### ✅ Verify Installation

After installation, verify everything works:

```bash
# Test Whisper
python -c "import whisper; print('✅ Whisper installed correctly')"

# Test CUDA (for NVIDIA GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output with RTX 3090:**
```
✅ Whisper installed correctly
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

#### ⚠️ Important Notes

- **Whisper is OPTIONAL**: The app works with Google Speech Recognition alone if Whisper doesn't install
- **ffmpeg is ESSENTIAL for Whisper**: Without ffmpeg, Whisper won't work
- **PyTorch with CUDA**: Required to leverage GPU. Use correct URL for your CUDA version
- **Git required**: To install Whisper from GitHub, you need Git installed. [Download here](https://git-scm.com/download/win)

### 3. Configure LM Studio

1. **Download and install** [LM Studio](https://lmstudio.ai/)
2. **Download a model**:
   - Go to "Search" or "Download" section
   - Search for a model (e.g., Llama 3, Mistral, Phi, etc.)
   - Click "Download"
3. **Start the server**:
   - Go to "Local Server" section
   - Select the downloaded model
   - Click "Start Server"
   - Verify it's on port **1234** (default)
   - You should see: `Server running at http://localhost:1234`

## ▶️ Usage

### Launch the application

```bash
python vocal_chat.py
```

The web interface will **automatically** open in your default browser at `http://localhost:7860`

### 🌐 Language Selection

The UI automatically detects your browser language:
- **Italian browser**: Interface in Italian
- **Other languages**: Interface in English

You can manually change language in the UI using the "🌐 UI Language" dropdown.

### 🎤 Configure Voice Recognition

#### Option 1: Google Speech Recognition (Default)
- ✅ Already active, no configuration needed
- ✅ Fast and reliable
- ❌ Requires internet connection
- ❌ Audio data sent to Google

#### Option 2: Whisper (Recommended for Privacy) ⭐
1. **Install Whisper** (if not already done):
   ```bash
   pip install git+https://github.com/openai/whisper.git
   ```

2. In the interface, go to **"🎤 Voice Recognition"** section

3. Select **"whisper - Local"**

4. Choose the **Whisper model**:
   - `tiny` (39 MB) - Very fast, basic accuracy
   - `base` (74 MB) - **Recommended for CPU** - great compromise
   - `small` (244 MB) - More accurate
   - `medium` (769 MB) - **Recommended for GPU** - excellent quality/speed
   - `large` (1.5 GB) - **Maximum accuracy** - perfect for RTX 3090!

5. Click **"📥 Load Model"** (first load downloads model from internet, then everything is local)

6. Wait for confirmation message ✅

**Whisper advantages:**
- 🔒 **Total privacy**: everything local, no data sent online
- 🎯 **More accurate**: better recognition of accents, dialects, noise
- 🌍 **99+ languages**: includes Italian, English, Spanish, French, German, Portuguese, Russian, Japanese, Chinese, Arabic and many more
- 📝 **Automatic punctuation**: adds commas and periods
- 🔊 **Robust**: works better with imperfect audio

**Model caching:**
- First time: Downloads model (~100MB-1.5GB)
- Saved in: `~/.cache/whisper/`
- Subsequent times: Instant loading from cache ✅
- No re-download needed!

### 🎯 How to interact

#### Method 1: Voice Input
1. Click the **microphone** icon 🎤
2. **Speak clearly** into the microphone
3. Click **STOP** ⏹️
4. ⏳ *The app transcribes automatically...*
5. ✅ **Text appears in the editable field**
6. 📝 **You can edit/correct** the text if needed
7. Click **"📤 Send"** to send to LM Studio
8. 🤖 Receive response (written and/or voiced)

**Advantages:**
- ✅ See what was understood before sending
- ✅ Correct transcription errors
- ✅ Add details or modifications
- ✅ Combine audio + manual editing

#### Method 2: Text Input
1. **Write** your message in the text box
2. Click **"📤 Send"**
3. Receive response (voice or text according to settings)

### ⚙️ Settings Panel

#### 🎛️ Model Parameters
- **Model**: Select which LM Studio model to use
- **Temperature** (0-2): 
  - `0.0-0.3`: Precise and deterministic responses
  - `0.4-0.7`: Balanced (recommended)
  - `0.8-2.0`: Creative and varied
- **Max Tokens** (50-2000): Maximum response length

#### 📝 System Prompt
Customize assistant behavior:
```
Examples:
- "You are a programming expert who explains simply"
- "You are a formal assistant for work emails"
- "You are a patient tutor teaching mathematics"
```

#### 🔊 Voice Settings
- **Voice**: Choose from installed system voices
- **Speed**: 50-300 (default: 150 words/minute)
- **Volume**: 0.0-1.0 (default: 0.9)

#### 🌍 Language
Change language for voice recognition (the model will respond in the language you write/speak)

### 🎮 Quick Controls

| Button | Function |
|--------|----------|
| 📤 Send | Send message to LM Studio |
| ⏹️ Stop Audio | Interrupt voice playback |
| 🗑️ Clear Chat | Delete conversation |
| 💾 Save JSON | Export in structured JSON format |
| 📄 Export TXT | Export in readable text format |
| 📝 Generate Summary | Create summary markdown document (NEW!) |

### 💾 Saving Conversations

Conversations are saved with timestamp:
- **JSON**: `conversation_20250125_143022.json`
- **TXT**: `conversation_20250125_143022.txt`
- **Markdown Summary**: `summary_20250125_143022.md` 🆕
- **DOCX Summary**: `summary_20250125_143022.docx` 🆕

Files are saved in the same folder as the program.

## 🎯 How it works

### **Conversation Flow:**
1. Click microphone 🎤
2. Speak
3. Click STOP ⏹️
4. ⏳ App transcribes automatically...
5. ✅ Text appears in field (you can edit it!)
6. Click "Send" to send to LM Studio

✅ **Automatic transcription**: When you FINISH recording (press stop), audio is transcribed automatically  
✅ **Editable text field**: Transcription appears in text field  
✅ **You can modify**: Before sending, you can correct/modify text

#### **Complete flow:**
```
Record audio → Stop → Auto transcription → Edit text (optional) → Send → AI Response
```

You can:
- ✅ See what was understood before sending
- ✅ Correct transcription errors
- ✅ Modify or add details
- ✅ Use only audio, only text, or both

### **Persistence:**
1. Start app → **automatically loads** last session
2. Chat normally → **automatically saves** after each response
3. Close app → conversation saved
4. Reopen app → conversation restored! 🔄

### **Document Export:**
1. Click "📝 Generate Summary (MD + DOCX)"
2. Wait for AI processing
3. Get **2 files**:
   - `.md` for markdown editors
   - `.docx` for Word/LibreOffice
4. View preview in app

### **Whisper Cache:**
First time with `large`:
```
⬇️ Downloading large model...
✅ Download completed and cached (1.5 GB)
✅ Model 'large' ready on NVIDIA GeForce RTX 3090!
📁 Cache: C:\Users\YourName\.cache\whisper\
```

Subsequent times:
```
📦 Model large found in cache: C:\Users\YourName\.cache\whisper\large.pt
✅ Loaded from cache (1.5 GB)
✅ Model 'large' ready on NVIDIA GeForce RTX 3090!
```

## 📁 File Structure
```
LLM_Vocal_Chat/
├── vocal_chat.py
├── requirements.txt
├── sessions/
│   └── current_session.json    ← Persistence!
├── conversation_*.json
├── conversation_*.txt
├── summary_*.md                ← MD Export!
└── summary_*.docx              ← DOCX Export!
```

## 🔧 Advanced Configuration

### Change LM Studio port

If LM Studio uses a different port than 1234:

1. Go to **"🔧 Advanced Settings"** tab
2. Modify **"LM Studio Server URL"** field
3. Example: `http://localhost:5000/v1/chat/completions`
4. Or modify directly in code:

```python
LM_STUDIO_URL = "http://localhost:YOURPORT/v1/chat/completions"
```

### Share interface on local network

In the code, change:

```python
demo.launch(share=False, inbrowser=True)
```

To:

```python
demo.launch(share=True, inbrowser=True)  # Creates temporary public link
# Or
demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)  # Local network
```

## 🔍 Troubleshooting

### ❌ "Cannot connect to LM Studio"

**Possible causes:**
- LM Studio is not running
- Server not started in LM Studio
- Wrong port

**Solutions:**
1. Open LM Studio
2. Go to "Local Server"
3. Click "Start Server"
4. Use **"🔌 Test Connection"** button in Advanced Settings tab

### ❌ "Could not understand, try again"

**With Google Speech Recognition:**
- Unclear audio
- Background noise
- Microphone not working

**Solutions:**
- Speak more slowly and clearly
- Reduce ambient noise
- Check microphone settings
- **Try Whisper**: much more robust with imperfect audio!
- Use text input as alternative
- Verify browser microphone permissions

### ⚠️ Whisper is slow or uses too much RAM

**Causes:**
- Model too large for your hardware
- Slow CPU without GPU

**Solutions:**
- **With GPU**: Make sure you checked "🎮 Use GPU"!
- **Without GPU**: Use `tiny` or `base` model instead of `medium/large`
- Close other applications
- With **RTX 3090**: confidently use `large` - it will be very fast!

**Verify GPU:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Shows GPU name
```

### ❌ Whisper won't install

**Error: "Could not find a version that satisfies the requirement openai-whisper"**

**Cause**: Whisper is not on standard PyPI, must be installed from GitHub.

**Solution:**

**Windows:**
```bash
# 1. Verify you have Git installed
git --version
# If you don't have it: https://git-scm.com/download/win

# 2. Install ffmpeg (IMPORTANT!)
choco install ffmpeg
# Or manual: https://www.gyan.dev/ffmpeg/builds/

# 3. Verify ffmpeg
ffmpeg -version

# 4. Install Whisper from GitHub
pip install git+https://github.com/openai/whisper.git
```

**macOS:**
```bash
brew install ffmpeg
pip install git+https://github.com/openai/whisper.git
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
pip install git+https://github.com/openai/whisper.git
```

**If it continues to fail:**
The app works anyway with Google Speech Recognition! Whisper is optional.

### ⚠️ CUDA not detected (torch.cuda.is_available() = False)

**Possible causes:**
- PyTorch installed without CUDA support
- NVIDIA drivers not updated
- CUDA Toolkit not installed or incompatible

**Solutions:**

1. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Verify NVIDIA drivers:**
```bash
nvidia-smi
# You should see info about your RTX 3090
```

3. **If nvidia-smi doesn't work:**
   - Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
   - Restart PC

4. **Verify compatible CUDA version:**
```bash
nvidia-smi
# Look at "CUDA Version" in top right
# Install compatible PyTorch for that version
```

**Compatibility table:**
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

### ❌ Microphone not detected

**Windows:**
1. Settings → Privacy → Microphone
2. Enable microphone access for apps
3. Verify browser has permissions

**macOS:**
1. System Preferences → Security & Privacy → Microphone
2. Enable terminal/browser

**Linux:**
```bash
sudo usermod -a -G audio $USER
# Then restart
```

### ⚠️ PyAudio errors

If you continue to have problems:

**Verify installation:**
```bash
python -c "import pyaudio; print('PyAudio OK')"
```

**Completely reinstall:**
```bash
pip uninstall pyaudio
pip cache purge
# Then use one of the installation methods above
```

### 🐌 Slow responses

**Causes:**
- Model too large for your hardware
- Max tokens too high
- Temperature too high

**Solutions:**
- Use smaller models (7B instead of 13B/70B)
- Reduce max_tokens to 200-300
- Lower temperature to 0.3-0.5
- In LM Studio, enable GPU acceleration if available

### 🔊 No audio output

**Verify:**
1. Is **"🔊 Read response aloud"** checkbox active?
2. Is volume not at zero?
3. Do system speakers work?

**Voice test:**
```python
import pyttsx3
engine = pyttsx3.init()
engine.say("Audio test")
engine.runAndWait()
```

### ❌ "ImportError: No module named 'whisper'"

**Cause**: Whisper not installed correctly.

**Quick solution:**
```bash
# Make sure you have ffmpeg installed
ffmpeg -version

# Reinstall Whisper
pip uninstall whisper
pip install git+https://github.com/openai/whisper.git
```

### ⚠️ App starts but doesn't detect GPU

**Verify in interface:**
1. Go to **"🔧 Advanced Settings"** tab
2. Check **"🎮 GPU Info"** section
3. Should say: "✅ GPU Detected: NVIDIA GeForce RTX 3090"

**If it says "⚠️ GPU not detected":**
- Follow "CUDA not detected" section above
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

## 📊 Frequently Asked Questions (FAQ)

### I have an NVIDIA GPU, how do I maximize it?

**Step 1: Verify prerequisites**
```bash
# 1. Verify NVIDIA drivers
nvidia-smi

# 2. Verify CUDA in PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

**If CUDA is False, reinstall PyTorch:**
```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 2: Configure Whisper with GPU**
1. Install Whisper correctly (see installation section)
2. In interface, check **"🎮 Use GPU"**
3. Choose `large` model for maximum quality
4. Click **"📥 Load Model"**
5. You'll see: "✅ Whisper model 'large' loaded on NVIDIA GeForce RTX 3090!"

**Step 3: Configure LM Studio with GPU**
1. Open LM Studio → Settings → Hardware
2. Select your GPU (RTX 3090)
3. Enable "GPU Offload" at 100% (or as many layers as you want)
4. Restart server

**Step 4: Verify everything works**
- In app, go to **"🔧 Advanced Settings"**
- **"🎮 GPU Info"** section should show: "✅ GPU Detected: NVIDIA GeForce RTX 3090"

**Optimal RTX 3090 Setup:**
- ✅ Whisper: `large` model with GPU enabled
- ✅ LM Studio: 100% GPU offload
- ✅ PyTorch: with CUDA 11.8+
- 🚀 Result: Ultra-fast system (~2-3 sec per transcription)!

### Can I use other models besides LM Studio?

Yes! The code is compatible with any server implementing the OpenAI API. Just change the URL:
- **Ollama**: `http://localhost:11434/v1/chat/completions`
- **LocalAI**: `http://localhost:8080/v1/chat/completions`
- **OpenAI**: `https://api.openai.com/v1/chat/completions` (requires API key)

### Does voice recognition work offline?

**Google Speech Recognition**: No, requires internet.

**Whisper**: **Yes!** Completely offline after initial model download.
- **First time**: downloads model from internet (~100MB-1.5GB depending on model)
- **Model saved in**: `~/.cache/whisper/` (Windows: `C:\Users\YourName\.cache\whisper\`)
- **After first download**: 100% local, no connection needed
- **Persistent cache**: If you download `base`, it stays downloaded forever!
- Privacy guaranteed: no data leaves your computer

### Which Whisper model should I choose?

| Model | Size | CPU Speed | GPU Speed | Accuracy | Recommended for |
|-------|------|-----------|-----------|----------|-----------------|
| tiny | 39 MB | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Old PCs, quick tests |
| base | 74 MB | ⚡⚡⚡⚡ | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | **CPU - GENERAL USE** ✅ |
| small | 244 MB | ⚡⚡⚡ | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Powerful PCs |
| medium | 769 MB | ⚡⚡ | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **GPU - RECOMMENDED** 🎮 |
| large | 1.5 GB | ⚡ | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **Powerful GPUs (RTX 3090)** 🚀 |

**Recommendations:**
- **With CPU**: Use `base` (great compromise)
- **With GPU RTX 2060-3060**: Use `medium` 
- **With GPU RTX 3070-4090**: Use `large` (maximum quality!)
- **RTX 3090 with 24GB VRAM**: `large` is super fast! (~2-3 seconds)

### How much VRAM is needed for Whisper?

| Model | Required VRAM | Notes |
|-------|---------------|-------|
| tiny | ~1 GB | Works everywhere |
| base | ~1 GB | Great for all GPUs |
| small | ~2 GB | GTX 1060 6GB and higher |
| medium | ~5 GB | RTX 2060 and higher |
| large | ~10 GB | **RTX 3080/3090/4090** ✅ |

**Your RTX 3090 with 24GB VRAM**: You can confidently use `large`!

### Can I use GPU for acceleration?

Yes! Configure GPU acceleration in LM Studio:
1. Settings → Hardware
2. Select GPU
3. Restart server

**For Whisper**: The app automatically detects GPU and uses it if available!

### How does the summary document work?

1. At the end of the conversation, click **"📝 Generate Summary Document"**
2. The AI ​​analyzes the entire conversation
3. It automatically extracts:
- Main topic
- Key points of discussion
- Decisions made
- Action items
- Technical details
4. Generates a well-formatted Markdown document
5. Saves it as `.md` and displays a preview
6. You can open it with any Markdown editor or viewer

### Where are the files saved?

All files are saved in the **same folder** where `vocal_chat.py` is located:
- **JSON Conversations**: `conversation_YYYYMMDD_HHMMSS.json`
- **TXT Conversations**: `conversation_YYYYMMDD_HHMMSS.txt`
- **MD Summaries**: `summary_YYYYMMDD_HHMMSS.md`

**Whisper Models** are saved in:
- Windows: `C:\Users\YourName\.cache\whisper\`
- macOS/Linux: `~/.cache/whisper/`

### Is it possible to use the OpenAI API instead of LM Studio?

Yes, just change:
```python
LM_STUDIO_URL = "https://api.openai.com/v1/chat/completions"

# And add to the request header:
headers = {
"Authorization": f"Bearer {OPENAI_API_KEY}",
"Content Type": "application/json"
}
```

### Are conversations private?

- **LM Studio**: ✅ Completely local and private
- **Whisper**: ✅ 100% local after downloading the model
- **Google Voice Recognition**: ❌ Audio data is sent to Google
- **Full Privacy Alternative**: Use Whisper + LM Studio = Zero online data!

## 📝 Technical Notes

###Architecture

```
┌────────────┐
│  Browser   │ ← Gradio Interface (Dark Mode)
└─────┬──────┘
      │
┌─────▼────────┐
│  Voice Chat  │ ← Python Script
│  .py         │
└─┬─────────┬──┘
  │         │
┌─▼─────┐ ┌─▼─────────┐
│Whisper│ │ LM Studio │
│ GPU   │ │ (Local)   │
│ RTX   │ │           │
└───────┘ └───────────┘
```

### Dependencies

| Library | Version | Purpose |
|----------|---------|-------|
| grade | 4.x+ | Web Interface |
| Speech Recognition | 3.x+ | Google Speech Recognition (fallback) |
| whisper | last | Local GPU Speech Recognition |
| pyttsx3 | 2.x+ | Text-to-Speech (TTS) |
| piaudio | 0.2.x | Audio Handling |
| requests | 2.x+ | Call API |
| mindless | 1.x+ | Audio Array Processing |
| scipy | 1.x+ | Audio Resampling for Whisper |
| Torch | 2.x+ | PyTorch Backend for Whisper (with CUDA for GPUs) |
| ffmpeg | 4.x+ | Audio Codec (external dependency for Whisper) |

### GPU Requirements

**For Whisper with GPUs:**
- Updated NVIDIA drivers (version 450.80+)
- CUDA Toolkit 11.8 or 12.1
- PyTorch with CUDA support
- GPU with at least 4GB VRAM (RTX 2060+)
- Optimal: RTX 3080/3090/4090 with 10GB+ VRAM

### Note on Limitations

- Whisper may be slow on older CPUs (use `tiny` or `base` model)
- For maximum GPU performance, use properly configured CUDA
- pyttsx3 may experience latency on Windows
- Gradio requires a modern browser (no IE)
- Max context tokens depends on the LM Studio model loaded
- ffmpeg is required for Whisper (external dependency)
- First time with Whisper: download model from the internet (then everything offline)

## 🎯Future Development Roadmap

- [x] ~~Whisper Support for Offline Recognition~~ ✅ Implemented!
- [x] ~~GPU Acceleration for Whisper~~ ✅ Implemented!
- [x] ~~AI Summary Document in Markdown~~ ✅ Implemented!
- [x] ~~Persistent Conversation Memory Across Sessions~~ ✅ Implemented!
- [ ] Formatted PDF Export
- [ ] Image Support (Vision Models)
- [ ] Extension Plugin System
- [ ] Multi-user Mode
- [ ] Advanced Conversation Analytics
- [ ] Integration with Other Services (Notion, Google Docs)
- [ ] Real-time Audio Streaming
- [ ] Trigger Word Detection ("Hey Claude")
- [ ] Whisper Fine-Tuning for Specific Voices

## 🤝 Contribute

Contributions, issues, and feature requests are welcome!

## 📦Project Files

```
voice-chat-lmstudio/
├── voice_chat.py # Main script
├── requirements.txt # Python dependencies
├── README.md # This guide
├── conversation_*.json # Saved conversations (generated by the app)
├── conversation_*.txt # Export text (generated by the app)
└── summary_*.md # AI summaries (generated by the app)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [LM Studio](https://lmstudio.ai/) - For the excellent local AI client
- [Gradio](https://gradio.app/) - For the framework UI
- [Google Speech Recognition](https://cloud.google.com/speech-to-text) - For speech recognition
- [pyttsx3](https://pyttsx3.readthedocs.io/) - For text-to-speech

## 📧Support

For issues or questions:
1. Check the "Troubleshooting" section
2. Check the FAQ
3. Open an issue on GitHub

---

**Have fun with your AI voice chat! 🚀**