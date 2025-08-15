import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import time
from scipy.ndimage import gaussian_filter1d
import colorsys
import imageio
import os
import tempfile
import soundfile
import audioread

# Configurazione pagina
st.set_page_config(
    page_title="Audio Visual Pattern Generator",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Audio Visual Pattern Generator")
st.markdown("Carica un brano musicale e guarda i pattern astratti generati dalle frequenze!")

class PatternGenerator:
    def __init__(self, audio_features):
        self.audio_features = audio_features
        self.colors = self.generate_random_colors()
        
    def generate_random_colors(self):
        """Genera una palette di colori casuali ogni volta"""
        base_hue = random.random()
        colors = []
        for i in range(6):
            hue = (base_hue + i * 0.15) % 1.0
            saturation = random.uniform(0.6, 1.0)
            lightness = random.uniform(0.3, 0.8)
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)
        return colors
    
    def pattern_1_glitch_blocks(self, frame_idx, width=400, height=300):
        """Pattern 1: Blocchi colorati glitch come nella prima immagine"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        num_blocks_x = random.randint(15, 40)
        num_blocks_y = random.randint(10, 25)
        
        block_width = width // num_blocks_x
        block_height = height // num_blocks_y
        
        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                freq_idx = (i + j) % len(freq_data)
                intensity = freq_data[freq_idx]
                
                if intensity < 0.3 and random.random() < 0.4:
                    continue
                
                actual_width = int(block_width * (0.5 + intensity * 0.5))
                actual_height = int(block_height * (0.5 + intensity * 0.5))
                
                x_offset = random.randint(-5, 5) if intensity > 0.7 else 0
                y_offset = random.randint(-3, 3) if intensity > 0.6 else 0
                
                x_start = max(0, i * block_width + x_offset)
                y_start = max(0, j * block_height + y_offset)
                x_end = min(width, x_start + actual_width)
                y_end = min(height, y_start + actual_height)
                
                color_idx = int((intensity + i/num_blocks_x + j/num_blocks_y) * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                
                brightness = 0.4 + intensity * 0.6
                final_color = [c * brightness for c in color]
                
                if x_start < x_end and y_start < y_end:
                    pattern[y_start:y_end, x_start:x_end] = final_color
                    
        return pattern
    
    def pattern_2_horizontal_stripes_glitch(self, frame_idx, width=400, height=300):
        """Pattern 2: Strisce orizzontali con glitch digitale"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        stripe_height = random.randint(1, 4)
        
        y = 0
        stripe_idx = 0
        
        while y < height:
            freq_idx = stripe_idx % len(freq_data)
            intensity = freq_data[freq_idx]
            
            current_stripe_height = stripe_height
            if intensity > 0.8:
                current_stripe_height = random.randint(1, 8)
            
            if intensity > 0.5:
                stripe_width = width
                x_start = 0
            else:
                stripe_width = int(width * (0.3 + intensity * 0.7))
                x_start = random.randint(0, max(1, width - stripe_width))
            
            if intensity > 0.7 and random.random() < 0.3:
                x_offset = random.randint(-20, 20)
                x_start = max(0, min(width - stripe_width, x_start + x_offset))
            
            color_idx = int((stripe_idx * 0.1 + intensity) * len(self.colors)) % len(self.colors)
            base_color = self.colors[color_idx]
            
            color = list(base_color)
            if intensity > 0.6 and random.random() < 0.2:
                color[random.randint(0, 2)] = random.random()
            
            brightness = 0.3 + intensity * 0.7
            final_color = [c * brightness for c in color]
            
            y_end = min(height, y + current_stripe_height)
            x_end = min(width, x_start + stripe_width)
            
            if x_start < x_end and y < y_end:
                pattern[y:y_end, x_start:x_end] = final_color
            
            y += current_stripe_height
            stripe_idx += 1
            
        return pattern
    
    def pattern_3_curved_flowing_lines(self, frame_idx, width=400, height=300):
        """Pattern 3: Linee curve fluide"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        num_curves = int(8 + np.mean(freq_data) * 15)
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        for curve_idx in range(num_curves):
            freq_idx = curve_idx % len(freq_data)
            intensity = freq_data[freq_idx]
            
            if intensity < 0.2:
                continue
            
            amplitude = height * 0.3 * intensity
            frequency = (curve_idx + 1) * 0.02
            phase = frame_idx * 0.1 + curve_idx * 0.5
            
            center_y = height * (curve_idx / num_curves)
            
            curve_y = center_y + amplitude * np.sin(x_coords * frequency + phase)
            
            line_thickness = max(1, int(5 * intensity))
            
            color_idx = curve_idx % len(self.colors)
            base_color = self.colors[color_idx]
            
            color_variation = 0.8 + intensity * 0.4
            final_color = [c * color_variation for c in base_color]
            
            for x in range(width):
                curve_center = int(curve_y[0, x])
                
                for thickness in range(-line_thickness//2, line_thickness//2 + 1):
                    y_pos = curve_center + thickness
                    
                    if 0 <= y_pos < height:
                        alpha = 1.0 - abs(thickness) / (line_thickness/2 + 1)
                        alpha *= intensity
                        
                        for c in range(3):
                            pattern[y_pos, x, c] = max(pattern[y_pos, x, c], 
                                                     final_color[c] * alpha)
        
        if frame_idx > 0:
            for c in range(3):
                pattern[:, :, c] = gaussian_filter1d(pattern[:, :, c], 
                                                   sigma=0.5, axis=1)
        
        return pattern

def extract_audio_features(audio_file):
    """Estrae le caratteristiche audio per la visualizzazione"""
    try:
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        
        if len(y) == 0:
            st.error("Il file audio sembra essere vuoto o corrotto")
            return None
            
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        n_freq_bins = min(50, magnitude.shape[0])
        
        spectral_features = []
        step = max(1, magnitude.shape[1] // 1000)
        
        for frame in range(0, magnitude.shape[1], step):
            frame_data = magnitude[:n_freq_bins, frame]
            if np.max(frame_data) > 0:
                frame_data = frame_data / np.max(frame_data)
            else:
                frame_data = np.zeros(n_freq_bins)
            spectral_features.append(frame_data)
        
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            'spectral_features': spectral_features,
            'tempo': float(tempo),
            'beats': beats,
            'duration': len(y) / sr,
            'sample_rate': sr
        }
    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {str(e)}")
        st.info("Suggerimento: prova con un file WAV o MP3 più piccolo")
        return None

def create_video_from_patterns(audio_features, pattern_type, fps=30, duration_seconds=15):
    """Genera un video MP4 animato a partire dai pattern usando imageio."""
    generator = PatternGenerator(audio_features)
    
    total_frames = int(fps * duration_seconds)
    max_audio_frames = len(audio_features['spectral_features'])
    if max_audio_frames < total_frames:
        st.warning(f"Durata audio insufficiente. Il video avrà una durata massima di {max_audio_frames/fps:.2f} secondi.")
        total_frames = max_audio_frames
    
    video_path = os.path.join(tempfile.gettempdir(), f"video_{int(time.time())}.mp4")

    width, height = 400, 300
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264', macro_block_size=1)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for frame_idx in range(total_frames):
        if pattern_type == "Blocchi Glitch":
            pattern = generator.pattern_1_glitch_blocks(frame_idx, width, height)
        elif pattern_type == "Strisce Orizzontali":
            pattern = generator.pattern_2_horizontal_stripes_glitch(frame_idx, width, height)
        else:
            pattern = generator.pattern_3_curved_flowing_lines(frame_idx, width, height)
        
        frame_rgb = (pattern * 255).astype(np.uint8)
        writer.append_data(frame_rgb)

        progress = (frame_idx + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Creazione video in corso: {int(progress * 100)}%")
        
    writer.close()
    
    progress_bar.empty()
    status_text.empty()
    
    return video_path

# Interfaccia Streamlit
uploaded_file = st.file_uploader(
    "Carica un file audio (MP3, WAV, M4A)", 
    type=['mp3', 'wav', 'm4a', 'flac']
)

if uploaded_file is not None:
    try:
        temp_audio_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File audio caricato con successo!")
        
        with st.spinner("Analizzando l'audio..."):
            audio_features = extract_audio_features(temp_audio_path)
            
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        audio_features = None
    
    if audio_features is not None:
        st.success("Analisi audio completata!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{audio_features['duration']:.2f}s")
        with col2:
            tempo_val = float(audio_features['tempo']) if audio_features['tempo'] is not None else 0.0
            st.metric("Tempo", f"{tempo_val:.1f} BPM")
        with col3:
            st.metric("Sample Rate", f"{audio_features['sample_rate']} Hz")
        
        st.subheader("🎬 Genera Video Animato")
        
        pattern_type = st.selectbox(
            "Seleziona il tipo di pattern:",
            ["Blocchi Glitch", "Strisce Orizzontali", "Linee Curve Fluide"]
        )
        
        video_duration = st.slider(
            "Durata video (secondi)", 
            min_value=5, 
            max_value=30, 
            value=15
        )
        video_fps = st.selectbox(
            "FPS Video", 
            [24, 30, 60], 
            index=1
        )
        
        if st.button("🎬 Genera Video MP4"):
            with st.spinner("Generando video... Questo può richiedere alcuni minuti."):
                try:
                    video_path = create_video_from_patterns(
                        audio_features, 
                        pattern_type, 
                        fps=video_fps, 
                        duration_seconds=video_duration
                    )
                    
                    if video_path and os.path.exists(video_path):
                        with open(video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        if len(video_bytes) > 0:
                            st.success("✅ Video generato con successo!")
                            st.video(video_bytes)
                            
                            st.download_button(
                                label="📥 Scarica Video MP4",
                                data=video_bytes,
                                file_name=f"pattern_{pattern_type.replace(' ', '_')}_{int(time.time())}.mp4",
                                mime="video/mp4"
                            )
                        else:
                            st.error("Il file video è vuoto")
                    else:
                        st.error("Impossibile creare il file video")
                        
                except Exception as e:
                    st.error(f"Errore durante la generazione: {str(e)}")
                    st.info("Prova con una durata più breve o un FPS più basso")
                    
        with st.expander("ℹ️ Come Funziona"):
            st.markdown("""
            **Pattern Generati:**
            
            1. **Blocchi Glitch**: Blocchi colorati di diverse dimensioni con effetti di glitch digitale
            2. **Strisce Orizzontali**: Linee orizzontali sottili con distorsioni e spostamenti
            3. **Linee Curve Fluide**: Curve sinusoidali che scorrono fluidamente seguendo l'audio
            
            **Caratteristiche:**
            - Colori casuali generati ad ogni esecuzione
            - Sincronizzazione con le frequenze audio
            - Effetti di glitch e distorsione
            - Pattern mai identici tra esecuzioni diverse
            - **Esportazione in Video MP4**
            """)

# Sidebar con informazioni
with st.sidebar:
    st.markdown("### 🎵 Audio Visual Generator")
    st.markdown("""
    Questa app analizza i brani musicali e genera pattern astratti 
    sincronizzati con le frequenze audio.
    
    **Features:**
    - 3 tipi di pattern differenti
    - Colori sempre casuali
    - Sincronizzazione tempo-reale
    - Effetti visuali dinamici
    - **Esportazione Video**
    """)
    
    st.markdown("### 🚀 Deploy su Streamlit")
    st.markdown("""
    Per deployare su Streamlit Cloud:
    1. Carica il codice su GitHub
    2. Aggiungi `requirements.txt` e `packages.txt`
    3. Connetti il repo a Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("💡 *I pattern sono generati in tempo reale e sono sempre unici!*")
