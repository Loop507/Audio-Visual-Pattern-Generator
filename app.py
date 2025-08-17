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
import subprocess
from PIL import Image, ImageDraw, ImageFont

# Configurazione pagina
st.set_page_config(
    page_title="Audio Visual Pattern Generator",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Audio Visual Pattern Generator by loop507")
st.markdown("Carica un brano musicale e guarda i pattern astratti generati dalle frequenze!")

class PatternGenerator:
    def __init__(self, audio_features, user_params):
        self.audio_features = audio_features
        self.colors = user_params.get("colors", self.generate_random_colors())
        self.background_color = user_params.get("background_color", (0, 0, 0))
        self.master_intensity = user_params.get("master_intensity", 1.0)
        self.glitch_effect = user_params.get("glitch_effect", 0.5)
        self.thickness = user_params.get("thickness", 0.5)
        
        # Calcola il volume RMS per ogni frame per una migliore sincronizzazione
        self.volume_levels = self._calculate_volume_levels()
        
    def _calculate_volume_levels(self):
        """Calcola i livelli di volume RMS per ogni frame per una migliore sincronizzazione"""
        try:
            y, sr = librosa.load(self.audio_features['audio_path'], sr=None, mono=True)
            hop_length = 512
            frame_length = 2048
            
            # Calcola RMS per ogni frame
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Normalizza i valori RMS
            if np.max(rms) > 0:
                rms = rms / np.max(rms)
            
            # Ridimensiona per adattarsi al numero di frame spettrali
            target_length = len(self.audio_features['spectral_features'])
            if len(rms) != target_length:
                # Interpola per adattare la lunghezza
                from scipy.interpolate import interp1d
                old_indices = np.linspace(0, 1, len(rms))
                new_indices = np.linspace(0, 1, target_length)
                f = interp1d(old_indices, rms, kind='linear', fill_value='extrapolate')
                rms = f(new_indices)
            
            return rms
        except Exception as e:
            # Fallback: usa la media spettrale
            return np.array([np.mean(frame) for frame in self.audio_features['spectral_features']])
        
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
    
    def get_intensity(self, frame_idx, base_intensity=1.0):
        """Migliorata sincronizzazione audio-video usando RMS e dati spettrali"""
        frame_idx = frame_idx % len(self.audio_features['spectral_features'])
        
        # Combina RMS (volume generale) e dati spettrali (frequenze)
        volume_intensity = self.volume_levels[frame_idx]
        freq_data = self.audio_features['spectral_features'][frame_idx]
        spectral_intensity = np.mean(freq_data)
        
        # Peso maggiore al volume RMS per una migliore reattività
        combined_intensity = (volume_intensity * 0.7 + spectral_intensity * 0.3) * base_intensity * self.master_intensity
        
        # Aggiungi un po' di smoothing per evitare cambi troppo bruschi
        if frame_idx > 0:
            prev_frame = (frame_idx - 1) % len(self.audio_features['spectral_features'])
            prev_volume = self.volume_levels[prev_frame]
            prev_spectral = np.mean(self.audio_features['spectral_features'][prev_frame])
            prev_combined = (prev_volume * 0.7 + prev_spectral * 0.3) * base_intensity * self.master_intensity
            
            # Leggero smoothing
            combined_intensity = combined_intensity * 0.8 + prev_combined * 0.2
        
        return combined_intensity

    def pattern_1_glitch_blocks(self, frame_idx, width, height):
        """Pattern 1: Blocchi colorati glitch - con migliore sincronizzazione"""
        pattern = np.zeros((height, width, 3))
        
        # Imposta lo sfondo
        pattern[:, :, 0] = self.background_color[0]
        pattern[:, :, 1] = self.background_color[1]
        pattern[:, :, 2] = self.background_color[2]
        
        frame_idx = frame_idx % len(self.audio_features['spectral_features'])
        freq_data = self.audio_features['spectral_features'][frame_idx]
        volume_level = self.volume_levels[frame_idx]
        
        # Adatta il numero di blocchi al volume
        base_blocks_x = int(15 + 25 * self.thickness)
        base_blocks_y = int(10 + 15 * self.thickness)
        
        # Riduci i blocchi se il volume è basso
        num_blocks_x = max(5, int(base_blocks_x * (0.3 + volume_level * 0.7)))
        num_blocks_y = max(3, int(base_blocks_y * (0.3 + volume_level * 0.7)))
        
        block_width = width // num_blocks_x
        block_height = height // num_blocks_y
        
        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                freq_idx = (i + j) % len(freq_data)
                intensity = freq_data[freq_idx] * volume_level  # Combina frequenza e volume
                
                # Sincronizzazione migliorata: blocchi molto meno visibili con audio basso
                if volume_level < 0.1 and random.random() > 0.3:
                    continue
                if intensity < 0.05 and random.random() > (1 - self.glitch_effect * volume_level):
                    continue
                
                # La dimensione dei blocchi dipende sia dall'intensità che dal volume
                size_factor = intensity * volume_level
                actual_width = int(block_width * (0.2 + size_factor * 0.8))
                actual_height = int(block_height * (0.2 + size_factor * 0.8))
                
                # Effetto glitch proporzionale al volume
                glitch_threshold = 0.7 * volume_level
                if intensity > glitch_threshold and random.random() < self.glitch_effect * volume_level:
                    x_offset = random.randint(-int(8 * volume_level), int(8 * volume_level))
                    y_offset = random.randint(-int(5 * volume_level), int(5 * volume_level))
                else:
                    x_offset = y_offset = 0
                
                x_start = max(0, i * block_width + x_offset)
                y_start = max(0, j * block_height + y_offset)
                x_end = min(width, x_start + actual_width)
                y_end = min(height, y_start + actual_height)
                
                color_idx = int((intensity + i/num_blocks_x + j/num_blocks_y) * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                
                # Luminosità basata su intensità e volume
                brightness = 0.2 + (intensity * volume_level) * 0.8 * self.master_intensity
                final_color = [c * brightness for c in color]
                
                if x_start < x_end and y_start < y_end:
                    pattern[y_start:y_end, x_start:x_end] = final_color
                    
        return pattern
    
    def pattern_2_horizontal_stripes_glitch(self, frame_idx, width, height):
        """Pattern 2: Strisce orizzontali con glitch digitale - con migliore sincronizzazione"""
        pattern = np.zeros((height, width, 3))
        
        # Imposta lo sfondo
        pattern[:, :, 0] = self.background_color[0]
        pattern[:, :, 1] = self.background_color[1]
        pattern[:, :, 2] = self.background_color[2]
        
        frame_idx = frame_idx % len(self.audio_features['spectral_features'])
        freq_data = self.audio_features['spectral_features'][frame_idx]
        volume_level = self.volume_levels[frame_idx]
        
        # Adatta lo spessore delle strisce al volume
        base_stripe_height = int(max(1, 4 * self.thickness))
        stripe_height = max(1, int(base_stripe_height * (0.5 + volume_level * 0.5)))
        
        y = 0
        stripe_idx = 0
        
        while y < height:
            freq_idx = stripe_idx % len(freq_data)
            intensity = freq_data[freq_idx] * volume_level  # Combina frequenza e volume
            
            # Salta le strisce se il volume è troppo basso
            if volume_level < 0.1 and random.random() > 0.4:
                y += stripe_height
                stripe_idx += 1
                continue
            
            current_stripe_height = stripe_height
            if intensity * volume_level > 0.8 * self.master_intensity:
                current_stripe_height = random.randint(1, int(8 * volume_level))
            
            # Larghezza delle strisce basata su intensità e volume
            if intensity * volume_level > 0.5 * self.master_intensity:
                stripe_width = width
                x_start = 0
            else:
                min_width_factor = 0.1 + volume_level * 0.2
                stripe_width = int(width * (min_width_factor + intensity * volume_level * 0.7))
                x_start = random.randint(0, max(1, width - stripe_width))
            
            # Effetto glitch proporzionale al volume
            if intensity > 0.7 and random.random() < self.glitch_effect * volume_level:
                x_offset = random.randint(-int(20 * volume_level), int(20 * volume_level))
                x_start = max(0, min(width - stripe_width, x_start + x_offset))
            
            color_idx = int((stripe_idx * 0.1 + intensity) * len(self.colors)) % len(self.colors)
            base_color = self.colors[color_idx]
            
            color = list(base_color)
            if intensity > 0.6 and random.random() < self.glitch_effect * volume_level:
                color[random.randint(0, 2)] = random.random()
            
            # Luminosità basata su intensità e volume
            brightness = 0.1 + (intensity * volume_level) * 0.9 * self.master_intensity
            final_color = [c * brightness for c in color]
            
            y_end = min(height, y + current_stripe_height)
            x_end = min(width, x_start + stripe_width)
            
            if x_start < x_end and y < y_end:
                pattern[y:y_end, x_start:x_end] = final_color
            
            y += current_stripe_height
            stripe_idx += 1
            
        return pattern
    
    def pattern_3_curved_flowing_lines(self, frame_idx, width, height):
        """Pattern 3: Linee curve fluide - con migliore sincronizzazione"""
        pattern = np.zeros((height, width, 3))
        
        # Imposta lo sfondo
        pattern[:, :, 0] = self.background_color[0]
        pattern[:, :, 1] = self.background_color[1]
        pattern[:, :, 2] = self.background_color[2]
        
        frame_idx = frame_idx % len(self.audio_features['spectral_features'])
        freq_data = self.audio_features['spectral_features'][frame_idx]
        volume_level = self.volume_levels[frame_idx]
        
        # Numero di curve basato su intensità spettrale e volume
        base_curves = int(8 + np.mean(freq_data) * 15)
        num_curves = max(2, int(base_curves * (0.3 + volume_level * 0.7)))
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        for curve_idx in range(num_curves):
            freq_idx = curve_idx % len(freq_data)
            intensity = freq_data[freq_idx] * volume_level  # Combina frequenza e volume
            
            # Salta le curve se il volume è troppo basso
            if volume_level < 0.1 and random.random() > 0.3:
                continue
            if intensity < 0.1 * self.master_intensity:
                continue
            
            # Ampiezza delle curve basata su intensità e volume
            amplitude = height * 0.3 * intensity * volume_level * self.master_intensity
            frequency = (curve_idx + 1) * 0.02 * (1 + volume_level * 0.5)
            phase = frame_idx * 0.1 * volume_level + curve_idx * 0.5 * self.glitch_effect
            
            center_y = height * (curve_idx / num_curves)
            
            curve_y = center_y + amplitude * np.sin(x_coords * frequency + phase)
            
            # Spessore delle linee basato su intensità e volume
            line_thickness = max(1, int(5 * intensity * volume_level * self.thickness))
            
            color_idx = curve_idx % len(self.colors)
            base_color = self.colors[color_idx]
            
            color_variation = 0.5 + (intensity * volume_level) * 0.5 * self.glitch_effect
            final_color = [c * color_variation for c in base_color]
            
            for x in range(width):
                curve_center = int(curve_y[0, x])
                
                for thickness in range(-line_thickness//2, line_thickness//2 + 1):
                    y_pos = curve_center + thickness
                    
                    if 0 <= y_pos < height:
                        alpha = 1.0 - abs(thickness) / (line_thickness/2 + 1)
                        alpha *= intensity * volume_level
                        
                        for c in range(3):
                            pattern[y_pos, x, c] = max(pattern[y_pos, x, c], 
                                                     final_color[c] * alpha)
        
        # Applica smoothing solo se c'è abbastanza attività
        if frame_idx > 0 and volume_level > 0.2:
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
            'sample_rate': sr,
            'audio_path': audio_file
        }
    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {str(e)}")
        st.info("Suggerimento: prova con un file WAV o MP3 più piccolo")
        return None

def create_video_with_audio(audio_path, video_path_no_audio, final_video_path):
    """
    Combina un file video (senza audio) con un file audio usando FFmpeg.
    """
    try:
        command = [
            'ffmpeg',
            '-y', 
            '-i', video_path_no_audio,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            final_video_path
        ]
        
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Errore nella combinazione di video e audio: {e.stderr}")
        return False

# Inizializza session_state per i colori se non esiste
if 'user_colors' not in st.session_state:
    st.session_state.user_colors = []
    st.session_state.num_colors = 4

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
        
        st.subheader("🎬 Controlli per la Generazione Video")
        
        col_select, col_slider = st.columns(2)
        with col_select:
            pattern_type = st.selectbox(
                "Seleziona il tipo di pattern:",
                ["Blocchi Glitch", "Strisce Orizzontali", "Linee Curve Fluide"]
            )
        
        with col_slider:
            aspect_ratio = st.selectbox(
                "Seleziona l'aspect ratio (proporzioni):",
                ["1:1 (Square)", "9:16 (Verticale)", "16:9 (Orizzontale)"]
            )
        
        # Nuovi controlli master
        st.subheader("⚙️ Controlli Master degli Effetti")
        
        col_controls = st.columns(3)
        with col_controls[0]:
            master_intensity = st.slider(
                "Intensità Master",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                help="Controlla la reattività dei pattern ai cambiamenti di volume."
            )
        with col_controls[1]:
            glitch_effect = st.slider(
                "Effetto Glitch/Random",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Regola l'intensità degli effetti casuali e di distorsione."
            )
        with col_controls[2]:
            thickness = st.slider(
                "Spessore Elementi",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                help="Controlla lo spessore delle linee, delle strisce o dei blocchi."
            )
        
        # Scelta dei colori
        st.markdown("---")
        st.subheader("🎨 Personalizza i Colori")
        
        col_bg, col_palette = st.columns(2)
        
        with col_bg:
            hex_bg = st.color_picker("Colore Sfondo", "#000000", key="bg_picker")
            bg_color_rgb = tuple(int(hex_bg[j:j+2], 16) / 255.0 for j in (1, 3, 5))

        with col_palette:
            custom_colors = st.checkbox("Usa colori personalizzati")
            user_colors = []
            if custom_colors:
                num_colors = st.slider("Numero di colori", 2, 8, st.session_state.num_colors, key='num_colors_slider')
                st.session_state.num_colors = num_colors
                if len(st.session_state.user_colors) != num_colors:
                    st.session_state.user_colors = [("#%06x" % random.randint(0, 0xFFFFFF)) for _ in range(num_colors)]
                
                cols_color_picker = st.columns(num_colors)
                for i in range(num_colors):
                    with cols_color_picker[i]:
                        hex_color = st.color_picker(f"Colore {i+1}", st.session_state.user_colors[i], key=f"color_picker_{i}")
                        st.session_state.user_colors[i] = hex_color
                user_colors = [tuple(int(c[j:j+2], 16) / 255.0 for j in (1, 3, 5)) for c in st.session_state.user_colors]
            else:
                user_colors = None
            
        # Titolo video
        st.markdown("---")
        st.subheader("✏️ Aggiungi un Titolo")
        video_title = st.text_input("Inserisci il titolo del video (lascia vuoto per non aggiungerlo)", "")
        
        if video_title:
            col_title_pos_v, col_title_pos_h = st.columns(2)
            with col_title_pos_v:
                title_position_v = st.selectbox(
                    "Posizione Verticale:",
                    ["In Alto", "In Basso"]
                )
            with col_title_pos_h:
                title_position_h = st.selectbox(
                    "Posizione Orizzontale:",
                    ["A Sinistra", "Centrato", "A Destra"]
                )
        else:
            title_position_v = None
            title_position_h = None

        if st.button("🎬 Genera Video MP4"):
            with st.spinner("Generando video... Questo può richiedere alcuni minuti."):
                try:
                    # Imposta dimensioni in base all'aspect ratio
                    if aspect_ratio == "1:1 (Square)":
                        width, height = 1080, 1080
                    elif aspect_ratio == "9:16 (Verticale)":
                        width, height = 720, 1280
                    else: # 16:9
                        width, height = 1280, 720
                    
                    # Genera il video senza audio usando imageio
                    video_no_audio_path = os.path.join(tempfile.gettempdir(), f"video_no_audio_{int(time.time())}.mp4")
                    
                    user_params = {
                        "master_intensity": master_intensity,
                        "glitch_effect": glitch_effect,
                        "thickness": thickness,
                        "colors": user_colors,
                        "background_color": bg_color_rgb
                    }
                    generator = PatternGenerator(audio_features, user_params)
                    total_frames = int(audio_features['duration'] * 30) # 30 FPS
                    
                    writer = imageio.get_writer(video_no_audio_path, fps=30, codec='libx264', macro_block_size=1)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Preparazione font per il titolo
                    if video_title:
                        try:
                            # Font per il titolo del video
                            title_font = ImageFont.truetype("arial.ttf", 40)
                        except IOError:
                            title_font = ImageFont.load_default()
                            
                    for frame_idx in range(total_frames):
                        if pattern_type == "Blocchi Glitch":
                            pattern = generator.pattern_1_glitch_blocks(frame_idx, width, height)
                        elif pattern_type == "Strisce Orizzontali":
                            pattern = generator.pattern_2_horizontal_stripes_glitch(frame_idx, width, height)
                        else:
                            pattern = generator.pattern_3_curved_flowing_lines(frame_idx, width, height)
                        
                        # Converte in uint8 e aggiungi il titolo con firma
                        frame_rgb = (pattern * 255).astype(np.uint8)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        if video_title:
                            draw = ImageDraw.Draw(pil_img)
                            
                            # Calcola la posizione del titolo principale
                            title_bbox = draw.textbbox((0, 0), video_title, font=title_font)
                            title_w = title_bbox[2] - title_bbox[0]
                            title_h = title_bbox[3] - title_bbox[1]
                            
                            padding = 20
                            title_x, title_y = 0, 0

                            if title_position_v == "In Alto":
                                title_y = padding
                            elif title_position_v == "In Basso":
                                title_y = height - title_h - padding
                            
                            if title_position_h == "A Sinistra":
                                title_x = padding
                            elif title_position_h == "A Destra":
                                title_x = width - title_w - padding
                            else: # Centrato orizzontalmente di default
                                title_x = (width - title_w) / 2
                            
                            # Disegna solo il titolo del video
                            draw.text((title_x, title_y), video_title, font=title_font, 
                                    fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                        
                        writer.append_data(np.array(pil_img))
                        
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Creazione video in corso: {int(progress * 100)}%")
                        
                    writer.close()
                    
                    # Combina il video con l'audio usando FFmpeg
                    final_video_path = os.path.join(tempfile.gettempdir(), f"final_video_{int(time.time())}.mp4")
                    
                    status_text.text("Combinando video e audio...")
                    
                    if create_video_with_audio(audio_features['audio_path'], video_no_audio_path, final_video_path):
                        if os.path.exists(final_video_path):
                            with open(final_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            st.success("✅ Video generato con successo!")
                            st.video(video_bytes)
                            
                            st.download_button(
                                label="📥 Scarica Video MP4",
                                data=video_bytes,
                                file_name=f"pattern_{pattern_type.replace(' ', '_')}.mp4",
                                mime="video/mp4"
                            )
                        else:
                            st.error("Il file video finale non è stato creato.")
                    
                    # Pulisci i file temporanei
                    os.remove(video_no_audio_path)
                    os.remove(temp_audio_path)
                    if os.path.exists(final_video_path):
                        os.remove(final_video_path)
                        
                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    st.error(f"Errore durante la generazione: {str(e)}")
                    st.info("Prova a ricaricare l'app e riprovare con un file diverso.")
