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

st.title("🎵 Audio Visual Pattern Generator")
st.markdown("Carica un brano musicale e guarda i pattern astratti generati dalle frequenze!")

class PatternGenerator:
    def __init__(self, audio_features, user_params):
        self.audio_features = audio_features
        self.colors = user_params.get("colors", self.generate_random_colors())
        self.background_color = user_params.get("background_color", (0, 0, 0))
        self.master_intensity = user_params.get("master_intensity", 1.0)
        self.glitch_effect = user_params.get("glitch_effect", 0.5)
        self.thickness = user_params.get("thickness", 0.5)
        
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
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        # Usa la media per una reattività più fluida
        average_intensity = np.mean(freq_data) * base_intensity * self.master_intensity
        return average_intensity

    def pattern_1_glitch_blocks(self, frame_idx, width, height):
        """Pattern 1: Blocchi colorati glitch come nella prima immagine"""
        pattern = np.zeros((height, width, 3))
        
        # Imposta lo sfondo
        pattern[:, :, 0] = self.background_color[0]
        pattern[:, :, 1] = self.background_color[1]
        pattern[:, :, 2] = self.background_color[2]
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        num_blocks_x = int(15 + 25 * self.thickness)
        num_blocks_y = int(10 + 15 * self.thickness)
        
        block_width = width // num_blocks_x
        block_height = height // num_blocks_y
        
        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                freq_idx = (i + j) % len(freq_data)
                intensity = freq_data[freq_idx]
                
                # Sincronizzazione con il volume: blocchi meno visibili con audio basso
                if intensity < 0.1 and random.random() > (1 - self.glitch_effect):
                    continue
                
                actual_width = int(block_width * (0.5 + intensity * 0.5))
                actual_height = int(block_height * (0.5 + intensity * 0.5))
                
                # Effetto glitch controllato
                x_offset = random.randint(-5, 5) if intensity > 0.7 * self.glitch_effect else 0
                y_offset = random.randint(-3, 3) if intensity > 0.6 * self.glitch_effect else 0
                
                x_start = max(0, i * block_width + x_offset)
                y_start = max(0, j * block_height + y_offset)
                x_end = min(width, x_start + actual_width)
                y_end = min(height, y_start + actual_height)
                
                color_idx = int((intensity + i/num_blocks_x + j/num_blocks_y) * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                
                brightness = 0.4 + intensity * 0.6 * self.master_intensity
                final_color = [c * brightness for c in color]
                
                if x_start < x_end and y_start < y_end:
                    pattern[y_start:y_end, x_start:x_end] = final_color
                    
        return pattern
    
    def pattern_2_horizontal_stripes_glitch(self, frame_idx, width, height):
        """Pattern 2: Strisce orizzontali con glitch digitale"""
        pattern = np.zeros((height, width, 3))
        
        # Imposta lo sfondo
        pattern[:, :, 0] = self.background_color[0]
        pattern[:, :, 1] = self.background_color[1]
        pattern[:, :, 2] = self.background_color[2]
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        stripe_height = int(max(1, 4 * self.thickness))
        
        y = 0
        stripe_idx = 0
        
        while y < height:
            freq_idx = stripe_idx % len(freq_data)
            intensity = freq_data[freq_idx]
            
            current_stripe_height = stripe_height
            if intensity > 0.8 * self.master_intensity:
                current_stripe_height = random.randint(1, 8)
            
            if intensity > 0.5 * self.master_intensity:
                stripe_width = width
                x_start = 0
            else:
                stripe_width = int(width * (0.3 + intensity * 0.7))
                x_start = random.randint(0, max(1, width - stripe_width))
            
            # Effetto glitch controllato
            if intensity > 0.7 and random.random() < self.glitch_effect:
                x_offset = random.randint(-20, 20)
                x_start = max(0, min(width - stripe_width, x_start + x_offset))
            
            color_idx = int((stripe_idx * 0.1 + intensity) * len(self.colors)) % len(self.colors)
            base_color = self.colors[color_idx]
            
            color = list(base_color)
            if intensity > 0.6 and random.random() < self.glitch_effect:
                color[random.randint(0, 2)] = random.random()
            
            brightness = 0.3 + intensity * 0.7 * self.master_intensity
            final_color = [c * brightness for c in color]
            
            y_end = min(height, y + current_stripe_height)
            x_end = min(width, x_start + stripe_width)
            
            if x_start < x_end and y < y_end:
                pattern[y:y_end, x_start:x_end] = final_color
            
            y += current_stripe_height
            stripe_idx += 1
            
        return pattern
    
    def pattern_3_curved_flowing_lines(self, frame_idx, width, height):
        """Pattern 3: Linee curve fluide"""
        pattern = np.zeros((height, width, 3))
        
        # Imposta lo sfondo
        pattern[:, :, 0] = self.background_color[0]
        pattern[:, :, 1] = self.background_color[1]
        pattern[:, :, 2] = self.background_color[2]
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        num_curves = int(8 + np.mean(freq_data) * 15)
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        for curve_idx in range(num_curves):
            freq_idx = curve_idx % len(freq_data)
            intensity = freq_data[freq_idx]
            
            if intensity < 0.2 * self.master_intensity:
                continue
            
            amplitude = height * 0.3 * intensity * self.master_intensity
            frequency = (curve_idx + 1) * 0.02
            phase = frame_idx * 0.1 + curve_idx * 0.5 * self.glitch_effect
            
            center_y = height * (curve_idx / num_curves)
            
            curve_y = center_y + amplitude * np.sin(x_coords * frequency + phase)
            
            line_thickness = max(1, int(5 * intensity * self.thickness))
            
            color_idx = curve_idx % len(self.colors)
            base_color = self.colors[color_idx]
            
            color_variation = 0.8 + intensity * 0.4 * self.glitch_effect
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
        
    def pattern_4_monoscope(self, frame_idx, width, height):
        """Pattern 4: Effetto Monoscopio con reattività audio"""
        # Crea un'immagine PIL vuota con il colore di sfondo
        pil_img = Image.new("RGB", (width, height), 
                             tuple(int(c * 255) for c in self.background_color))
        draw = ImageDraw.Draw(pil_img)

        # Calcola l'intensità media per il frame corrente
        intensity = self.get_intensity(frame_idx)
        
        # Effetto glitch
        glitch_shift = 0
        if random.random() < self.glitch_effect:
            glitch_shift = random.randint(-5, 5)

        # Disegna il cerchio principale
        circle_radius = int(width * (0.3 + 0.2 * intensity * self.master_intensity))
        x_center, y_center = width // 2 + glitch_shift, height // 2 + glitch_shift
        draw.ellipse(
            (x_center - circle_radius, y_center - circle_radius,
             x_center + circle_radius, y_center + circle_radius),
            outline="white",
            width=max(2, int(8 * self.thickness))
        )

        # Disegna le linee a croce
        line_thickness = max(2, int(6 * self.thickness))
        draw.line(
            (x_center, 0, x_center, height), 
            fill="white", width=line_thickness
        )
        draw.line(
            (0, y_center, width, y_center), 
            fill="white", width=line_thickness
        )

        # Disegna le barre colorate
        bar_height = height // 8
        num_colors_to_show = len(self.colors)
        bar_width = width // num_colors_to_show
        
        for i in range(num_colors_to_show):
            color = self.colors[i]
            x0 = i * bar_width
            y0 = height - bar_height
            x1 = x0 + bar_width
            y1 = height
            
            # Fai reagire le barre all'intensità audio
            color_mod = [c * (0.5 + intensity * 0.5) for c in color]
            
            draw.rectangle(
                (x0, y0, x1, y1), 
                fill=tuple(int(c * 255) for c in color_mod)
            )

        # Disegna una griglia nel cerchio
        grid_spacing = int(30 * (1 - self.thickness))
        if grid_spacing > 0:
            for i in range(x_center - circle_radius, x_center + circle_radius, grid_spacing):
                draw.line((i, y_center - circle_radius, i, y_center + circle_radius), fill=(100, 100, 100))
            for i in range(y_center - circle_radius, y_center + circle_radius, grid_spacing):
                draw.line((x_center - circle_radius, i, x_center + circle_radius, i), fill=(100, 100, 100))

        # Converte l'immagine PIL in un array NumPy
        return np.array(pil_img) / 255.0

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
                ["Blocchi Glitch", "Strisce Orizzontali", "Linee Curve Fluide", "Monoscopio"]
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
        st.subheader("✍️ Aggiungi un Titolo")
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
                            # Prova a caricare un font comune, altrimenti usa il default
                            font = ImageFont.truetype("arial.ttf", 40)
                        except IOError:
                            font = ImageFont.load_default()
                            
                    for frame_idx in range(total_frames):
                        if pattern_type == "Blocchi Glitch":
                            pattern = generator.pattern_1_glitch_blocks(frame_idx, width, height)
                        elif pattern_type == "Strisce Orizzontali":
                            pattern = generator.pattern_2_horizontal_stripes_glitch(frame_idx, width, height)
                        elif pattern_type == "Linee Curve Fluide":
                            pattern = generator.pattern_3_curved_flowing_lines(frame_idx, width, height)
                        else: # Monoscopio
                            pattern = generator.pattern_4_monoscope(frame_idx, width, height)
                        
                        # Converte in uint8 e aggiungi il titolo
                        frame_rgb = (pattern * 255).astype(np.uint8)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        if video_title:
                            draw = ImageDraw.Draw(pil_img)
                            # Calcola la posizione del testo
                            bbox = draw.textbbox((0, 0), video_title, font=font)
                            text_w = bbox[2] - bbox[0]
                            text_h = bbox[3] - bbox[1]
                            
                            padding = 20
                            x, y = 0, 0

                            if title_position_v == "In Alto":
                                y = padding
                            elif title_position_v == "In Basso":
                                y = height - text_h - padding
                            
                            if title_position_h == "A Sinistra":
                                x = padding
                            elif title_position_h == "A Destra":
                                x = width - text_w - padding
                            else: # Centrato orizzontalmente di default
                                x = (width - text_w) / 2
                            
                            draw.text((x, y), video_title, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                        
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
