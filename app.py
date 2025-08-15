import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import time
from scipy.ndimage import gaussian_filter1d
import colorsys
from PIL import Image
import io
import base64
import zipfile
import os

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
        """Pattern 1: Blocchi colorati glitch"""
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
        """Pattern 2: Strisce orizzontali con glitch"""
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
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        
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
            beats = onset_frames
        except Exception:
            tempo = 120.0
            beats = np.array([])
        
        return {
            'spectral_features': spectral_features,
            'tempo': float(tempo),
            'beats': beats,
            'duration': len(y) / sr,
            'sample_rate': sr
        }
    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {str(e)}")
        return None

def create_gif_simple(audio_features, pattern_type, num_frames=20):
    """
    Crea una GIF animata - METODO SEMPLICE GARANTITO
    """
    try:
        generator = PatternGenerator(audio_features)
        width, height = 400, 300
        
        # Crea le immagini
        images = []
        progress_bar = st.progress(0)
        
        for i in range(num_frames):
            progress_bar.progress(i / num_frames)
            
            audio_frame_idx = int(i * len(audio_features['spectral_features']) / num_frames)
            
            if pattern_type == "Blocchi Glitch":
                pattern = generator.pattern_1_glitch_blocks(audio_frame_idx, width, height)
            elif pattern_type == "Strisce Orizzontali":
                pattern = generator.pattern_2_horizontal_stripes_glitch(audio_frame_idx, width, height)
            else:
                pattern = generator.pattern_3_curved_flowing_lines(audio_frame_idx, width, height)
            
            # Converti in PIL Image
            pattern_uint8 = (np.clip(pattern, 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(pattern_uint8)
            images.append(img)
        
        progress_bar.empty()
        
        # Crea GIF in memoria
        gif_buffer = io.BytesIO()
        images[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=images[1:],
            duration=200,  # millisecondi per frame
            loop=0
        )
        gif_buffer.seek(0)
        
        return gif_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Errore nella creazione GIF: {e}")
        return None

def create_frame_sequence(audio_features, pattern_type, num_frames=30):
    """
    Crea una sequenza di immagini PNG - ALTERNATIVA AL VIDEO
    """
    try:
        generator = PatternGenerator(audio_features)
        width, height = 800, 600
        
        # Crea ZIP con le immagini
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            progress_bar = st.progress(0)
            
            for i in range(num_frames):
                progress_bar.progress(i / num_frames)
                
                audio_frame_idx = int(i * len(audio_features['spectral_features']) / num_frames)
                
                if pattern_type == "Blocchi Glitch":
                    pattern = generator.pattern_1_glitch_blocks(audio_frame_idx, width, height)
                elif pattern_type == "Strisce Orizzontali":
                    pattern = generator.pattern_2_horizontal_stripes_glitch(audio_frame_idx, width, height)
                else:
                    pattern = generator.pattern_3_curved_flowing_lines(audio_frame_idx, width, height)
                
                # Converti in PNG
                pattern_uint8 = (np.clip(pattern, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(pattern_uint8)
                
                # Salva nel ZIP
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                zip_file.writestr(f"frame_{i:04d}.png", img_buffer.getvalue())
            
            progress_bar.empty()
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Errore nella creazione sequenza: {e}")
        return None

# --- Interfaccia Streamlit ---
uploaded_file = st.file_uploader(
    "Carica un file audio (MP3, WAV, M4A)", 
    type=['mp3', 'wav', 'm4a', 'flac']
)

if uploaded_file is not None:
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File audio caricato con successo!")
        
        with st.spinner("Analizzando l'audio..."):
            audio_features = extract_audio_features("temp_audio.wav")
            
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        audio_features = None
    
    if audio_features is not None:
        st.success("Analisi audio completata!")
        
        # Mostra informazioni sull'audio
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{audio_features['duration']:.2f}s")
        with col2:
            tempo_val = float(audio_features['tempo']) if audio_features['tempo'] is not None else 0.0
            st.metric("Tempo", f"{tempo_val:.1f} BPM")
        with col3:
            st.metric("Sample Rate", f"{audio_features['sample_rate']} Hz")
        
        # Selezione del tipo di pattern
        pattern_type = st.selectbox(
            "Seleziona il tipo di pattern:",
            ["Blocchi Glitch", "Strisce Orizzontali", "Linee Curve Fluide"]
        )
        
        # Generazione dei pattern
        if st.button("🎨 Genera Visualizzazione"):
            with st.spinner("Generando i pattern visuali..."):
                
                generator = PatternGenerator(audio_features)
                
                # Anteprima statica
                st.subheader("Anteprima Pattern Generati")
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    with col:
                        frame_idx = i * len(audio_features['spectral_features']) // 3
                        
                        if pattern_type == "Blocchi Glitch":
                            pattern = generator.pattern_1_glitch_blocks(frame_idx)
                        elif pattern_type == "Strisce Orizzontali":
                            pattern = generator.pattern_2_horizontal_stripes_glitch(frame_idx)
                        else:
                            pattern = generator.pattern_3_curved_flowing_lines(frame_idx)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(pattern, aspect='auto')
                        ax.axis('off')
                        ax.set_title(f"Frame {frame_idx}")
                        
                        st.pyplot(fig)
                        plt.close()
                
                # Slider per esplorare i frame
                st.subheader("Esplora i Pattern nel Tempo")
                frame_slider = st.slider(
                    "Frame", 
                    0, 
                    len(audio_features['spectral_features']) - 1, 
                    0
                )
                
                # Mostra il frame selezionato
                if pattern_type == "Blocchi Glitch":
                    pattern = generator.pattern_1_glitch_blocks(frame_slider)
                elif pattern_type == "Strisce Orizzontali":
                    pattern = generator.pattern_2_horizontal_stripes_glitch(frame_slider)
                else:
                    pattern = generator.pattern_3_curved_flowing_lines(frame_slider)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(pattern, aspect='auto')
                ax.axis('off')
                ax.set_title(f"Pattern Frame {frame_slider} - {pattern_type}", fontsize=16)
                
                st.pyplot(fig)
                plt.close()
                
                # SEZIONE ANIMAZIONI - METODI ALTERNATIVI
                st.subheader("🎬 Esporta Animazioni")
                
                # Opzione 1: GIF Animata
                st.markdown("**🎞️ GIF Animata**")
                col1, col2 = st.columns(2)
                with col1:
                    gif_frames = st.slider("Numero frame GIF", 10, 50, 20)
                with col2:
                    if st.button("🎞️ Crea GIF Animata"):
                        with st.spinner("Creando GIF animata..."):
                            gif_data = create_gif_simple(audio_features, pattern_type, gif_frames)
                            
                            if gif_data:
                                st.success("✅ GIF creata con successo!")
                                st.image(gif_data)
                                st.download_button(
                                    label="📥 Scarica GIF",
                                    data=gif_data,
                                    file_name=f"pattern_{pattern_type.replace(' ', '_')}.gif",
                                    mime="image/gif"
                                )
                
                # Opzione 2: Sequenza di Immagini
                st.markdown("**📸 Sequenza Immagini PNG**")
                col1, col2 = st.columns(2)
                with col1:
                    png_frames = st.slider("Numero frame PNG", 20, 100, 30)
                with col2:
                    if st.button("📸 Crea Sequenza PNG"):
                        with st.spinner("Creando sequenza di immagini..."):
                            zip_data = create_frame_sequence(audio_features, pattern_type, png_frames)
                            
                            if zip_data:
                                st.success("✅ Sequenza creata con successo!")
                                st.info(f"📦 {png_frames} immagini PNG ad alta risoluzione (800x600)")
                                st.download_button(
                                    label="📥 Scarica ZIP con Immagini",
                                    data=zip_data,
                                    file_name=f"frames_{pattern_type.replace(' ', '_')}.zip",
                                    mime="application/zip"
                                )
                
                # Informazioni sui colori
                st.subheader("Palette Colori Generata")
                color_cols = st.columns(len(generator.colors))
                for i, (col, color) in enumerate(zip(color_cols, generator.colors)):
                    with col:
                        color_array = np.full((50, 50, 3), color)
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(color_array)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                        st.write(f"Colore {i+1}")
        
        # Informazioni tecniche
        with st.expander("ℹ️ Come Funziona"):
            st.markdown("""
            **Alternative al Video MP4:**
            
            1. **GIF Animata**: File leggero, compatibile ovunque, ideale per preview
            2. **Sequenza PNG**: Immagini ad alta qualità, puoi creare video con software esterni
            
            **Per creare video da PNG:**
            ```bash
            # Con FFmpeg:
            ffmpeg -r 30 -i frame_%04d.png -c:v libx264 output.mp4
            
            # Con After Effects, Premiere, DaVinci Resolve, ecc.
            ```
            
            **Pattern Generati:**
            - Sincronizzazione con frequenze audio
            - Colori casuali ad ogni esecuzione
            - Effetti glitch e animazioni fluide
            """)

# Sidebar
with st.sidebar:
    st.markdown("### 🎵 Audio Visual Generator")
    st.markdown("""
    **Alternative Video:**
    - 🎞️ GIF Animata (immediata)
    - 📸 Sequenza PNG (alta qualità)
    - Niente dipendenze video complesse!
    
    **Features:**
    - 3 tipi di pattern
    - Colori sempre diversi
    - Export multipli
    - Funziona sempre!
    """)

st.markdown("---")
st.markdown("💡 *Metodi alternativi che funzionano SEMPRE!*")
