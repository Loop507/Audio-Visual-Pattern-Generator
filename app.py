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
        self.colors = user_params.get("colors") or self.generate_random_colors()
        self.background_color = user_params.get("background_color", (0, 0, 0))
        self.master_intensity = user_params.get("master_intensity", 1.0)
        self.glitch_effect = user_params.get("glitch_effect", 0.5)
        self.thickness = user_params.get("thickness", 0.5)

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

    def get_intensity(self, audio_idx, base_intensity=1.0):
        freq_data = self.audio_features['spectral_features'][audio_idx % len(self.audio_features['spectral_features'])]
        return float(np.mean(freq_data)) * base_intensity * self.master_intensity

    def pattern_1_glitch_blocks(self, audio_idx, width, height):
        pattern = np.full((height, width, 3), self.background_color, dtype=np.float32)
        freq_data = self.audio_features['spectral_features'][audio_idx % len(self.audio_features['spectral_features'])]

        num_blocks_x = int(15 + 25 * self.thickness)
        num_blocks_y = int(10 + 15 * self.thickness)
        block_width = max(1, width // num_blocks_x)
        block_height = max(1, height // num_blocks_y)

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                freq_idx = (i + j) % len(freq_data)
                intensity = float(freq_data[freq_idx])

                if intensity < 0.1 and random.random() > (1 - self.glitch_effect):
                    continue

                actual_width = int(block_width * (0.5 + intensity * 0.5))
                actual_height = int(block_height * (0.5 + intensity * 0.5))

                x_offset = random.randint(-5, 5) if intensity > 0.7 * self.glitch_effect else 0
                y_offset = random.randint(-3, 3) if intensity > 0.6 * self.glitch_effect else 0

                x_start = max(0, i * block_width + x_offset)
                y_start = max(0, j * block_height + y_offset)
                x_end = min(width, x_start + actual_width)
                y_end = min(height, y_start + actual_height)

                color_idx = int((intensity + i / num_blocks_x + j / num_blocks_y) * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                brightness = 0.4 + intensity * 0.6 * self.master_intensity
                final_color = [c * brightness for c in color]

                if x_start < x_end and y_start < y_end:
                    pattern[y_start:y_end, x_start:x_end] = final_color

        return pattern

    def pattern_2_horizontal_stripes_glitch(self, audio_idx, width, height):
        pattern = np.full((height, width, 3), self.background_color, dtype=np.float32)
        freq_data = self.audio_features['spectral_features'][audio_idx % len(self.audio_features['spectral_features'])]

        stripe_height = int(max(1, 4 * self.thickness))
        y = 0
        stripe_idx = 0

        while y < height:
            freq_idx = stripe_idx % len(freq_data)
            intensity = float(freq_data[freq_idx])

            current_stripe_height = stripe_height
            if intensity > 0.8 * self.master_intensity:
                current_stripe_height = random.randint(1, 8)

            if intensity > 0.5 * self.master_intensity:
                stripe_width = width
                x_start = 0
            else:
                stripe_width = int(width * (0.3 + intensity * 0.7))
                x_start = random.randint(0, max(1, width - stripe_width))

            if intensity > 0.7 and random.random() < self.glitch_effect:
                x_offset = random.randint(-20, 20)
                x_start = max(0, min(width - stripe_width, x_start + x_offset))

            color_idx = int((stripe_idx * 0.1 + intensity) * len(self.colors)) % len(self.colors)
            base_color = list(self.colors[color_idx])

            if intensity > 0.6 and random.random() < self.glitch_effect:
                base_color[random.randint(0, 2)] = random.random()

            brightness = 0.3 + intensity * 0.7 * self.master_intensity
            final_color = [c * brightness for c in base_color]

            y_end = min(height, y + current_stripe_height)
            x_end = min(width, x_start + stripe_width)

            if x_start < x_end and y < y_end:
                pattern[y:y_end, x_start:x_end] = final_color

            y += current_stripe_height
            stripe_idx += 1

        return pattern

    def pattern_3_curved_flowing_lines(self, audio_idx, width, height):
        """Pattern vettorizzato: linee curve fluide - molto più veloce"""
        pattern = np.full((height, width, 3), self.background_color, dtype=np.float32)
        freq_data = self.audio_features['spectral_features'][audio_idx % len(self.audio_features['spectral_features'])]

        num_curves = int(8 + float(np.mean(freq_data)) * 15)
        x_coords = np.arange(width)

        for curve_idx in range(num_curves):
            freq_idx = curve_idx % len(freq_data)
            intensity = float(freq_data[freq_idx])

            if intensity < 0.2 * self.master_intensity:
                continue

            amplitude = height * 0.3 * intensity * self.master_intensity
            frequency = (curve_idx + 1) * 0.02
            phase = audio_idx * 0.1 + curve_idx * 0.5 * self.glitch_effect
            center_y = height * (curve_idx / num_curves)

            # Vettorizzato: calcola tutta la curva in una sola operazione
            curve_y = (center_y + amplitude * np.sin(x_coords * frequency + phase)).astype(int)

            line_thickness = max(1, int(5 * intensity * self.thickness))
            color_idx = curve_idx % len(self.colors)
            base_color = self.colors[color_idx]
            color_variation = 0.8 + intensity * 0.4 * self.glitch_effect
            final_color = np.array([c * color_variation for c in base_color], dtype=np.float32)

            # Disegna lo spessore con operazioni vettorizzate
            for t in range(-line_thickness // 2, line_thickness // 2 + 1):
                ys = np.clip(curve_y + t, 0, height - 1)
                denom = max(line_thickness / 2 + 1, 1e-6)
                alpha = (1.0 - abs(t) / denom) * intensity
                for c in range(3):
                    np.maximum(
                        pattern[ys, x_coords, c],
                        final_color[c] * alpha,
                        out=pattern[ys, x_coords, c]
                    )

        if audio_idx > 0:
            for c in range(3):
                pattern[:, :, c] = gaussian_filter1d(pattern[:, :, c], sigma=0.5, axis=1)

        return pattern


def extract_audio_features(audio_file):
    """Estrae le caratteristiche audio — robusto contro errori di shape e dtype"""
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
        for frame in range(magnitude.shape[1]):
            frame_data = magnitude[:n_freq_bins, frame].copy()
            max_val = float(np.max(frame_data))
            if max_val > 0:
                frame_data = frame_data / max_val
            else:
                frame_data = np.zeros(n_freq_bins, dtype=np.float32)
            spectral_features.append(frame_data.astype(np.float32))

        # Fix: beat_track restituisce array multidimensionali in versioni recenti di librosa
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr)
        # Garantisce che tempo sia uno scalare Python float
        tempo_val = float(np.atleast_1d(tempo_raw).flat[0])

        duration = float(len(y)) / float(sr)

        return {
            'spectral_features': spectral_features,
            'tempo': tempo_val,
            'beats': beats,
            'duration': duration,
            'sample_rate': int(sr),
            'audio_path': audio_file
        }

    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {str(e)}")
        st.info("Suggerimento: prova con un file WAV o MP3 più piccolo")
        return None


def find_font(size=60):
    """Cerca un font disponibile sul sistema"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def create_video_with_audio(audio_path, video_path_no_audio, final_video_path):
    """Combina video e audio con FFmpeg, con controllo disponibilità"""
    try:
        # Verifica che ffmpeg sia disponibile
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("FFmpeg non trovato. Installa FFmpeg sul server per combinare audio e video.")
        return False

    try:
        command = [
            'ffmpeg', '-y',
            '-i', video_path_no_audio,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            final_video_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Errore nella combinazione di video e audio: {e.stderr}")
        return False


def generate_preview_frame(audio_features, pattern_type, user_params, width=320, height=180):
    """Genera un singolo frame di anteprima a bassa risoluzione"""
    generator = PatternGenerator(audio_features, user_params)
    mid_idx = len(audio_features['spectral_features']) // 2

    if pattern_type == "Blocchi Glitch":
        pattern = generator.pattern_1_glitch_blocks(mid_idx, width, height)
    elif pattern_type == "Strisce Orizzontali":
        pattern = generator.pattern_2_horizontal_stripes_glitch(mid_idx, width, height)
    else:
        pattern = generator.pattern_3_curved_flowing_lines(mid_idx, width, height)

    frame_rgb = np.clip(pattern * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_rgb)


def generate_report_text(audio_features, pattern_type, aspect_ratio, user_params,
                         video_title="", filename=""):
    """Genera il report stilizzato da copiare/incollare per social/YouTube"""

    duration = audio_features['duration']
    tempo = audio_features['tempo']
    sr = audio_features['sample_rate']
    mins = int(duration // 60)
    secs = int(duration % 60)

    all_means = [float(np.mean(f)) for f in audio_features['spectral_features']]
    avg_energy = float(np.mean(all_means))

    # STYLE e PROCESS descrivono l'app, non il pattern
    style   = "Spectral Pattern Synthesis / Real-Time Frequency Mapping"
    process = "STFT Analysis → Spectral Normalization → Frame Synthesis → H.264 Encode"
    quote_en = "It's not a video. It's the music rendered as light."
    quote_it = "Non è un video. È la musica resa come luce."

    fn_clean = filename.replace(".mp4", "").upper() if filename else pattern_type.upper().replace(" ", "_")

    # Bit depth descrittivo basato su sample rate
    bit_info = "32-bit Float" if sr >= 44100 else "16-bit"

    # Clipping point basato su energia media
    if avg_energy > 0.65:
        clip_en, clip_it = "Clipping_Point", "Punto_di_Clipping"
    elif avg_energy > 0.35:
        clip_en, clip_it = "Near_Clipping", "Vicino_al_Clipping"
    else:
        clip_en, clip_it = "Sub_Clipping", "Sotto_Clipping"

    # Hashtag
    base_tags = (
        "#AudioVisual #GlitchArt #AlgorithmicVideo #SoundDesign "
        "#NewMediaArt #DataNoise #SignalProcessing #FrequencyArt #VisualMusic"
    )
    pattern_tags = {
        "Blocchi Glitch":      "#GlitchBlocks #Brutalist #PixelCollapse #ComputationalMinimalism #SignalCorruption #RecursiveCollapse",
        "Strisce Orizzontali": "#ChromaticScan #HorizontalGlitch #DigitalBrutalism #ColorDisplacement #StripeGlitch #ScanLines",
        "Linee Curve Fluide":  "#WaveformArt #SinusoidalFlow #OrganicSignal #FluidFrequency #SoundWave #FlowField",
    }
    all_tags = f"{base_tags} {pattern_tags.get(pattern_type, '')}"

    report = f"""[AVPG_ARCHIVE] // {fn_clean} // H.264 // AAC

:: STYLE: {style}
:: ENGINE: audio_visual_pattern_generator [02.00]
:: AUDIO: {sr // 1000}kHz / {bit_info} / {tempo:.1f} BPM / {clip_en}
:: PATTERN: {pattern_type}
:: PROCESS: {process}
:: DURATION: {mins:02d}:{secs:02d}

"{quote_en}"
> Direction & Algorithm: Loop507

{all_tags}



[AVPG_ARCHIVE] // {fn_clean} // H.264 // AAC

:: STILE: {style}
:: MOTORE: audio_visual_pattern_generator [02.00]
:: AUDIO: {sr // 1000} kHz / {bit_info} / {tempo:.1f} BPM / {clip_it}
:: PATTERN: {pattern_type}
:: PROCESSO: {process}
:: DURATA: {mins:02d}:{secs:02d}

"{quote_it}"
> Regia e Algoritmo: Loop507

{all_tags}"""

    return report.strip()


# ─── Session State ────────────────────────────────────────────────────────────
if 'user_colors' not in st.session_state:
    st.session_state.user_colors = []
    st.session_state.num_colors = 4
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
if 'video_filename' not in st.session_state:
    st.session_state.video_filename = None
if 'report_text' not in st.session_state:
    st.session_state.report_text = None
if 'report_filename' not in st.session_state:
    st.session_state.report_filename = None

# ─── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Carica un file audio (MP3, WAV, M4A, FLAC)",
    type=['mp3', 'wav', 'm4a', 'flac']
)

audio_features = None
temp_audio_path = None

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
            st.metric("Tempo", f"{audio_features['tempo']:.1f} BPM")
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
                "Seleziona l'aspect ratio:",
                ["1:1 (Square)", "9:16 (Verticale)", "16:9 (Orizzontale)"]
            )

        st.subheader("⚙️ Controlli Master degli Effetti")
        col_controls = st.columns(3)
        with col_controls[0]:
            master_intensity = st.slider("Intensità Master", 0.1, 2.0, 1.0,
                                         help="Controlla la reattività ai cambiamenti di volume.")
        with col_controls[1]:
            glitch_effect = st.slider("Effetto Glitch/Random", 0.0, 1.0, 0.5,
                                      help="Intensità degli effetti casuali e di distorsione.")
        with col_controls[2]:
            thickness = st.slider("Spessore Elementi", 0.1, 2.0, 0.5,
                                  help="Spessore delle linee, strisce o blocchi.")

        st.markdown("---")
        st.subheader("🎨 Personalizza i Colori")
        col_bg, col_palette = st.columns(2)

        with col_bg:
            hex_bg = st.color_picker("Colore Sfondo", "#000000", key="bg_picker")
            bg_color_rgb = tuple(int(hex_bg[j:j + 2], 16) / 255.0 for j in (1, 3, 5))

        with col_palette:
            custom_colors = st.checkbox("Usa colori personalizzati")
            user_colors = []
            if custom_colors:
                num_colors = st.slider("Numero di colori", 2, 8,
                                       st.session_state.num_colors, key='num_colors_slider')
                st.session_state.num_colors = num_colors
                if len(st.session_state.user_colors) != num_colors:
                    st.session_state.user_colors = [
                        "#%06x" % random.randint(0, 0xFFFFFF) for _ in range(num_colors)
                    ]
                cols_color_picker = st.columns(num_colors)
                for i in range(num_colors):
                    with cols_color_picker[i]:
                        hex_color = st.color_picker(f"Colore {i + 1}",
                                                    st.session_state.user_colors[i],
                                                    key=f"color_picker_{i}")
                        st.session_state.user_colors[i] = hex_color
                user_colors = [
                    tuple(int(c[j:j + 2], 16) / 255.0 for j in (1, 3, 5))
                    for c in st.session_state.user_colors
                ]
            else:
                user_colors = None

        st.markdown("---")
        st.subheader("✍️ Aggiungi un Titolo")
        video_title = st.text_input("Titolo del video (lascia vuoto per non aggiungerlo)", "")
        title_position_v = title_position_h = None
        if video_title:
            col_tv, col_th = st.columns(2)
            with col_tv:
                title_position_v = st.selectbox("Posizione Verticale:", ["In Alto", "In Basso"])
            with col_th:
                title_position_h = st.selectbox("Posizione Orizzontale:", ["A Sinistra", "Centrato", "A Destra"])

        # ─── Anteprima rapida ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🖼️ Anteprima (frame centrale, bassa risoluzione)")

        user_params_preview = {
            "master_intensity": master_intensity,
            "glitch_effect": glitch_effect,
            "thickness": thickness,
            "colors": user_colors,
            "background_color": bg_color_rgb
        }

        with st.spinner("Generando anteprima..."):
            preview_img = generate_preview_frame(audio_features, pattern_type, user_params_preview)
        st.image(preview_img, caption=f"Pattern: {pattern_type}", use_container_width=True)

        # ─── Generazione video ────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🎬 Genera Video MP4"):
            with st.spinner("Generando video... Questo può richiedere alcuni minuti."):
                try:
                    if aspect_ratio == "1:1 (Square)":
                        width, height = 1080, 1080
                    elif aspect_ratio == "9:16 (Verticale)":
                        width, height = 720, 1280
                    else:
                        width, height = 1280, 720

                    video_no_audio_path = os.path.join(
                        tempfile.gettempdir(), f"video_no_audio_{int(time.time())}.mp4"
                    )
                    final_video_path = os.path.join(
                        tempfile.gettempdir(), f"final_video_{int(time.time())}.mp4"
                    )

                    user_params = {
                        "master_intensity": master_intensity,
                        "glitch_effect": glitch_effect,
                        "thickness": thickness,
                        "colors": user_colors,
                        "background_color": bg_color_rgb
                    }
                    generator = PatternGenerator(audio_features, user_params)
                    total_frames = int(audio_features['duration'] * 30)
                    audio_frame_step = len(audio_features['spectral_features']) / max(total_frames, 1)

                    writer = imageio.get_writer(video_no_audio_path, fps=30,
                                               codec='libx264', macro_block_size=1)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    font_title = find_font(60) if video_title else None

                    for frame_idx in range(total_frames):
                        audio_idx = int(frame_idx * audio_frame_step)
                        if pattern_type == "Blocchi Glitch":
                            pattern = generator.pattern_1_glitch_blocks(audio_idx, width, height)
                        elif pattern_type == "Strisce Orizzontali":
                            pattern = generator.pattern_2_horizontal_stripes_glitch(audio_idx, width, height)
                        else:
                            pattern = generator.pattern_3_curved_flowing_lines(audio_idx, width, height)

                        frame_rgb = np.clip(pattern * 255, 0, 255).astype(np.uint8)
                        pil_img = Image.fromarray(frame_rgb)

                        if video_title and font_title:
                            draw = ImageDraw.Draw(pil_img)
                            bbox = draw.textbbox((0, 0), video_title, font=font_title)
                            text_w = bbox[2] - bbox[0]
                            text_h = bbox[3] - bbox[1]
                            padding = 40
                            x = padding if title_position_h == "A Sinistra" \
                                else (width - text_w - padding if title_position_h == "A Destra"
                                      else (width - text_w) / 2)
                            y = padding if title_position_v == "In Alto" \
                                else height - text_h - padding
                            draw.text((x, y), video_title, font=font_title,
                                      fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))

                        writer.append_data(np.array(pil_img))
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Creazione video: {int(progress * 100)}%")

                    writer.close()

                    status_text.text("Combinando video e audio...")
                    success = create_video_with_audio(temp_audio_path, video_no_audio_path, final_video_path)

                    if success and os.path.exists(final_video_path):
                        with open(final_video_path, 'rb') as vf:
                            st.session_state.video_bytes = vf.read()
                        video_filename = f"pattern_{pattern_type.replace(' ', '_')}.mp4"
                        st.session_state.video_filename = video_filename
                        st.session_state.report_text = generate_report_text(
                            audio_features=audio_features,
                            pattern_type=pattern_type,
                            aspect_ratio=aspect_ratio,
                            user_params=user_params,
                            video_title=video_title,
                            filename=video_filename
                        )
                        st.session_state.report_filename = f"report_{pattern_type.replace(' ', '_')}.txt"
                        st.success("✅ Video generato con successo!")
                    else:
                        st.error("Il file video finale non è stato creato.")

                    # Pulizia file temporanei
                    for path in [video_no_audio_path, final_video_path]:
                        if path and os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass

                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    st.error(f"Errore durante la generazione: {str(e)}")
                    st.info("Prova a ricaricare l'app e riprovare con un file diverso.")

        # ── Risultati persistenti (sopravvivono al rerun dei download button) ──
        if st.session_state.video_bytes is not None:
            st.markdown("---")
            st.video(st.session_state.video_bytes)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Scarica Video MP4",
                    data=st.session_state.video_bytes,
                    file_name=st.session_state.video_filename,
                    mime="video/mp4",
                    key="dl_video"
                )
            with col_dl2:
                st.download_button(
                    label="📄 Scarica Report .txt",
                    data=st.session_state.report_text.encode("utf-8"),
                    file_name=st.session_state.report_filename,
                    mime="text/plain",
                    key="dl_report"
                )

            st.markdown("---")
            st.subheader("📋 Report / Descrizione per Social & YouTube")
            st.code(st.session_state.report_text, language=None)
