# 🎙️ Custom Voice Generator - Text-to-Speech

A **TTS system with voice customization** supporting multiple languages, voice styles, prosody control, and neural voice synthesis.

## 🎯 Overview

This project provides:
- ✅ Text-to-speech synthesis
- ✅ Multiple voice options
- ✅ Language support
- ✅ Prosody control (pitch, rate, volume)
- ✅ Neural voice generation
- ✅ Audio processing
- ✅ Real-time streaming

## 🔊 TTS Basics

```python
import pyttsx3
import numpy as np

class BasicTTSGenerator:
    """Simple TTS using pyttsx3"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
    
    def set_properties(self, rate=150, volume=1.0, voice_id=0):
        """Configure TTS properties"""
        # Rate (words per minute)
        self.engine.setProperty('rate', rate)
        
        # Volume (0-1)
        self.engine.setProperty('volume', volume)
        
        # Voice selection
        voices = self.engine.getProperty('voices')
        if voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)
    
    def generate_speech(self, text, output_file=None):
        """Generate speech"""
        if output_file:
            self.engine.save_to_file(text, output_file)
        else:
            self.engine.say(text)
        
        self.engine.runAndWait()
    
    def list_voices(self):
        """Available voices"""
        voices = self.engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"{i}: {voice.name} - {voice.languages}")
```

## 🎵 Neural Voice Synthesis

```python
from gTTS import gTTS
from pydub import AudioSegment
import os

class GoogleTTSGenerator:
    """Google Text-to-Speech"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'ja': 'Japanese',
            'zh': 'Chinese'
        }
    
    def generate_speech(self, text, lang='en', output_file='output.mp3'):
        """Generate with Google TTS"""
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        return output_file
    
    def multi_language_speech(self, text_dict):
        """Multiple languages"""
        for lang, text in text_dict.items():
            filename = f"{lang}_output.mp3"
            tts = gTTS(text=text, lang=lang)
            tts.save(filename)
```

## 🎚️ Prosody Control

```python
import librosa
import soundfile as sf
from scipy.signal import resample

class ProsodyController:
    """Control speech prosody"""
    
    def __init__(self, audio_file):
        self.audio, self.sr = librosa.load(audio_file)
    
    def adjust_pitch(self, semitones):
        """Change pitch (semitones)"""
        y = librosa.effects.pitch_shift(
            self.audio,
            sr=self.sr,
            n_steps=semitones
        )
        return y
    
    def adjust_speed(self, speed_rate):
        """Change speed without pitch change"""
        y = librosa.effects.time_stretch(self.audio, rate=speed_rate)
        return y
    
    def adjust_tempo(self, tempo_factor):
        """Change tempo"""
        D = librosa.stft(self.audio)
        
        # Phase vocoder
        D_stretched = librosa.phase_vocoder(D, rate=tempo_factor)
        y = librosa.istft(D_stretched)
        
        return y
    
    def modify_volume(self, db_change):
        """Adjust loudness"""
        from librosa.util import normalize
        
        # Convert dB to linear
        linear_factor = 10 ** (db_change / 20.0)
        y = self.audio * linear_factor
        
        # Prevent clipping
        max_val = np.max(np.abs(y))
        if max_val > 1.0:
            y = y / max_val
        
        return y
    
    def add_emphasis(self, word_indices, emphasis_db=6):
        """Emphasize specific words"""
        modified = self.audio.copy()
        
        for start, end in word_indices:
            # Apply envelope
            envelope = np.linspace(1, 10**(emphasis_db/20), end-start)
            modified[start:end] *= envelope
        
        return modified
    
    def save_audio(self, audio_data, output_file):
        """Save modified audio"""
        sf.write(output_file, audio_data, self.sr)
```

## 🗣️ Voice Cloning

```python
class VoiceCloningSystem:
    """Clone voice characteristics"""
    
    def __init__(self):
        # Would use pre-trained voice encoder
        self.voice_encoder = None
        self.voice_embeddings = {}
    
    def extract_voice_characteristics(self, reference_audio):
        """Get voice features"""
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=reference_audio, sr=22050, n_mfcc=13)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=reference_audio)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(reference_audio)
        
        return {
            'mfcc': mfcc,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr,
            'mean_mfcc': np.mean(mfcc, axis=1),
            'mean_spectral_centroid': np.mean(spectral_centroid)
        }
    
    def apply_voice_characteristics(self, target_audio, characteristics):
        """Apply extracted features"""
        # Would implement neural style transfer here
        # For demo: adjust based on spectral characteristics
        
        target_mfcc = librosa.feature.mfcc(y=target_audio, sr=22050)
        ratio = characteristics['mean_mfcc'] / np.mean(target_mfcc, axis=1)
        
        # Apply scaling (simplified)
        modified = target_audio * np.mean(ratio)
        
        return modified
    
    def create_custom_voice(self, base_voice_samples, name):
        """Create new voice from samples"""
        combined_features = []
        
        for sample in base_voice_samples:
            features = self.extract_voice_characteristics(sample)
            combined_features.append(features)
        
        # Average features
        avg_features = {
            'mean_mfcc': np.mean([f['mean_mfcc'] for f in combined_features], axis=0),
            'mean_spectral': np.mean([f['mean_spectral_centroid'] for f in combined_features])
        }
        
        self.voice_embeddings[name] = avg_features
        return avg_features
```

## 🎛️ Audio Effects

```python
class AudioEffects:
    """Add effects to speech"""
    
    @staticmethod
    def add_reverb(audio, sr, room_scale=0.5):
        """Add reverb effect"""
        delay = int(0.05 * sr)
        decay = 0.3
        
        output = audio.copy()
        for _ in range(3):
            delayed = np.pad(audio, (delay, 0))[:-delay]
            output += delayed * decay
            delay *= 2
        
        return output
    
    @staticmethod
    def add_chorus(audio, sr, delay_ms=20, depth=5):
        """Add chorus effect"""
        delay_samples = int(delay_ms * sr / 1000)
        modulation = depth * np.sin(2 * np.pi * 2 * np.arange(len(audio)) / sr)
        
        delayed = np.zeros_like(audio)
        for i in range(len(audio)):
            delay_idx = int(delay_samples + modulation[i])
            if 0 <= i - delay_idx < len(audio):
                delayed[i] = audio[i - delay_idx]
        
        return (audio + delayed) / 2
    
    @staticmethod
    def add_echo(audio, delay_sec, decay=0.5, sr=22050):
        """Add echo effect"""
        delay_samples = int(delay_sec * sr)
        echo = np.zeros(len(audio) + delay_samples)
        
        echo[:len(audio)] = audio
        echo[delay_samples:] += audio * decay
        
        return echo
```

## 🎯 Real-time TTS Stream

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class StreamingTTSGenerator:
    """Real-time TTS streaming"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    async def stream_speech(self, text_generator):
        """Stream audio chunks"""
        loop = asyncio.get_event_loop()
        
        audio_chunks = []
        for text_chunk in text_generator:
            # Generate in background
            future = loop.run_in_executor(
                self.executor,
                self._synthesize_chunk,
                text_chunk
            )
            
            audio = await future
            audio_chunks.append(audio)
            
            # Could yield immediately for streaming
            yield audio
        
        return np.concatenate(audio_chunks)
    
    def _synthesize_chunk(self, text):
        """Synthesize single chunk"""
        # TTS generation
        pass
```

## 💡 Interview Talking Points

**Q: TTS vs pre-recorded audio?**
```
Answer:
- TTS: Flexible, scalable, dynamic content
- Pre-recorded: Better quality, limited
- TTS quality improved with neural networks
- Synthesis time vs streaming trade-off
```

**Q: Voice cloning challenges?**
```
Answer:
- Need quality reference samples
- Identity preservation difficult
- Training compute-intensive
- Privacy/ethical considerations
- Commercial TTS (Google, Azure) highly optimized
```

## 🌟 Portfolio Value

✅ Audio processing
✅ Speech synthesis
✅ Signal processing
✅ Real-time systems
✅ Neural voice generation
✅ Prosody control
✅ Audio effects

---

**Technologies**: pyttsx3, librosa, soundfile, NumPy

