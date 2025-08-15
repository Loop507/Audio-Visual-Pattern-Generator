import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter1d
import colorsys
from PIL import Image
import os
import tempfile
import random
import time

# ─────────────────────────────────────────────────────────────
# 🔥 DISABILITA NUMBA PER COMPATIBILITÀ CON PYTHON 3.13
os.environ['NUMBA_DISABLE_JIT'] = '1'
# ─────────────────────────────────────────────────────────────

# Ora importa librosa (senza paura di get_call_template)
import librosa

# --- Configurazione pagina ---
st.set_page_config(
    page_title="Audio Visual Pattern Generator",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Audio Visual Pattern Generator")
st.markdown("Carica un brano musicale e guarda i pattern astratti generati dalle frequenze!")

# --- Classe PatternGenerator ---
class PatternGenerator:
    def __init__(self, audio_features):
        self.audio_features = audio_features
        self.colors = self.generate_random_colors()
        
    def generate_random_colors(self):
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
                            pattern[y_pos, x, c] = max(pattern[y_pos, x, c], final_color[c] * alpha)
        
        if frame_idx > 0:
            for c in range(3):
                pattern[:, :, c] = gaussian_filter1d(pattern[:, :, c], sigma=0.5, axis=1)
        
        return pattern

# --- Estrazione features audio ---
def extract_audio_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        if len(y) == 0:
            st.error("Il file audio è vuoto o corrotto.")
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
        
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                intervals = np.diff(onset_times)
                if len(intervals) > 0:
                    avg_interval = np.median(intervals)
                    tempo = 60.0 / avg_interval if avg_interval > 0 else 120.0
                else:
                    tempo = 120.0
            else:
                tempo = 120.0
        except:
            tempo = 120.0
        
        return {
            'spectral_features': spectral_features,
            'tempo': float(tempo),
            'duration': len(y) / sr,
            'sample_rate': sr
        }
    except Exception as e:
        st.error(f"Errore nell'analisi audio: {str(e)}")
        return None

# --- Generazione video MP4 con imageio + ffmpeg ---
def create_video_from_patterns(audio_features, pattern_type, fps=30, duration_seconds=15):
    generator = PatternGenerator(audio_features)
    total_frames = len(audio_features['spectral_features'])
    
    target_frames = int(fps * duration_seconds)
    frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_path = temp_file.name
    temp_file.close()

    st.info(f"🎥 Inizio generazione video: {video_path}")
    st.info(f"📊 {target_frames} frame a {fps} FPS")

    try:
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            format='FFMPEG',
            mode='I',
            codec='libx264',
            pixelformat='yuv420p',
            output_params=['-crf', '23']
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, idx in enumerate(frame_indices):
            status_text.text(f"Rendering frame {i+1}/{target_frames}")
            progress_bar.progress((i + 1) / target_frames)

            if pattern_type == "Blocchi Glitch":
                pattern = generator.pattern_1_glitch_blocks(idx, width=400, height=300)
            elif pattern_type == "Strisce Orizzontali":
                pattern = generator.pattern_2_horizontal_stripes_glitch(idx, width=400, height=300)
            else:
                pattern = generator.pattern_3_curved_flowing_lines(idx, width=400, height=300)
            
            frame = (pattern * 255).astype(np.uint8)
            writer.append_data(frame)

        writer.close()
        st.success("✅ Video creato con successo!")

        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            file_size = os.path.getsize(video_path)
            st.info(f"💾 Dimensione file: {file_size:,} byte")
            return video_path
        else:
            st.error("❌ Il file video è vuoto o non esiste.")
            
    except Exception as e:
        st.error("❌ Errore durante la creazione del video")
        st.code(f"Errore: {str(e)}")
        import traceback
        st.text("Traceback:")
        st.code(traceback.format_exc())
        
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except:
                pass
    
    return None

# --- Interfaccia Streamlit ---
uploaded_file = st.file_uploader("Carica un file audio (MP3, WAV, M4A)", type=['mp3', 'wav', 'm4a', 'flac'])

if uploaded_file is not None:
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Audio caricato!")
        
        with st.spinner("🎧 Analisi in corso..."):
            audio_features = extract_audio_features("temp_audio.wav")
            
    except Exception as e:
        st.error(f"Errore nel caricamento: {str(e)}")
        audio_features = None
    
    if audio_features is not None:
        st.success("✅ Analisi completata!")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Durata", f"{audio_features['duration']:.1f}s")
        with col2: st.metric("Tempo", f"{audio_features['tempo']:.1f} BPM")
        with col3: st.metric("Sample Rate", f"{audio_features['sample_rate']} Hz")
        
        pattern_type = st.selectbox(
            "Tipo di pattern",
            ["Blocchi Glitch", "Strisce Orizzontali", "Linee Curve Fluide"]
        )
        
        if st.button("🎨 Genera Visualizzazione"):
            generator = PatternGenerator(audio_features)
            
            st.subheader("🖼️ Anteprima")
            cols = st.columns(3)
            for i, col in enumerate(cols):
                frame_idx = i * len(audio_features['spectral_features']) // 3
                if pattern_type == "Blocchi Glitch":
                    pattern = generator.pattern_1_glitch_blocks(frame_idx)
                elif pattern_type == "Strisce Orizzontali":
                    pattern = generator.pattern_2_horizontal_stripes_glitch(frame_idx)
                else:
                    pattern = generator.pattern_3_curved_flowing_lines(frame_idx)
                
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(pattern)
                ax.axis('off')
                ax.set_title(f"Frame {frame_idx}")
                col.pyplot(fig)
                plt.close()
            
            st.subheader("🎬 Genera Video MP4")
            duration = st.slider("Durata (s)", 5, 30, 10)
            fps = st.selectbox("FPS", [15, 24, 30], index=1)
            
            if st.button("🎥 Crea Video", key="gen_video"):
                video_path = create_video_from_patterns(
                    audio_features,
                    pattern_type,
                    fps=fps,
                    duration_seconds=duration
                )
                
                if video_path and os.path.exists(video_path):
                    with open(video_path, 'rb') as f:
                        video_bytes = f.read()
                    st.success("🎉 Video pronto per il download!")
                    st.download_button(
                        label="📥 Scarica Video MP4",
                        data=video_bytes,
                        file_name=f"pattern_{pattern_type.replace(' ', '_')}_{int(time.time())}.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("❌ Generazione video fallita.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🎵 Pattern Generator")
    st.markdown("""
    Crea animazioni astratte sincronizzate con la musica.
    
    - 3 stili diversi
    - Colori casuali
    - Esporta in MP4
    """)

st.markdown("---")
st.markdown("💡 *Ogni pattern è unico!*")
