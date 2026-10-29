import streamlit as st
import numpy as np
import librosa
import random
from scipy.ndimage import gaussian_filter1d
import colorsys
import imageio
import os
import shutil
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

PATTERN_NAMES = ["Blocchi Glitch", "Strisce Orizzontali", "Linee Curve Fluide"]


class PatternGenerator:
    """
    Generatore di pattern audio-reattivi.

    Riproducibilità: ogni frame usa un RNG dedicato derivato da (seed, audio_idx),
    quindi lo stesso seed + lo stesso brano + gli stessi parametri producono
    sempre lo stesso video, e l'anteprima coincide col frame corrispondente.
    Le dimensioni (spessori, jitter, frequenze) sono relative alla risoluzione.
    """

    def __init__(self, audio_features, user_params):
        self.audio_features = audio_features
        self.seed = int(user_params.get("seed", 0))
        self.colors = user_params.get("colors") or self.generate_random_colors()
        self.background_color = user_params.get("background_color", (0, 0, 0))
        self.master_intensity = user_params.get("master_intensity", 1.0)
        self.glitch_effect = user_params.get("glitch_effect", 0.5)
        self.thickness = user_params.get("thickness", 0.5)

    # ── utilità ──────────────────────────────────────────────────────────────
    def generate_random_colors(self):
        rng = random.Random(self.seed)
        base_hue = rng.random()
        colors = []
        for i in range(6):
            hue = (base_hue + i * 0.15) % 1.0
            saturation = rng.uniform(0.6, 1.0)
            lightness = rng.uniform(0.3, 0.8)
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors

    def _rng(self, audio_idx):
        return random.Random(self.seed * 1_000_003 + int(audio_idx))

    def _freq(self, audio_idx):
        feats = self.audio_features['spectral_features']
        return feats[int(audio_idx) % len(feats)]

    @staticmethod
    def _px(width, height):
        """Fattore di scala rispetto a un riferimento di 720 px sul lato corto."""
        return min(width, height) / 720.0

    def render_frame(self, pattern_type, audio_idx, width, height):
        if pattern_type == "Blocchi Glitch":
            return self.pattern_1_glitch_blocks(audio_idx, width, height)
        if pattern_type == "Strisce Orizzontali":
            return self.pattern_2_horizontal_stripes_glitch(audio_idx, width, height)
        return self.pattern_3_curved_flowing_lines(audio_idx, width, height)

    # ── pattern 1 ────────────────────────────────────────────────────────────
    def pattern_1_glitch_blocks(self, audio_idx, width, height):
        rng = self._rng(audio_idx)
        px = self._px(width, height)
        pattern = np.full((height, width, 3), self.background_color, dtype=np.float32)
        freq_data = self._freq(audio_idx)

        num_blocks_x = int(15 + 25 * self.thickness)
        num_blocks_y = int(10 + 15 * self.thickness)
        block_width = max(1, width // num_blocks_x)
        block_height = max(1, height // num_blocks_y)
        jitter_x = max(1, int(round(5 * px)))
        jitter_y = max(1, int(round(3 * px)))

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                intensity = float(freq_data[(i + j) % len(freq_data)])

                # Buchi casuali nei punti silenziosi, proporzionali al glitch
                if intensity < 0.1 and rng.random() < self.glitch_effect:
                    continue

                actual_width = max(1, int(block_width * (0.5 + intensity * 0.5)))
                actual_height = max(1, int(block_height * (0.5 + intensity * 0.5)))

                # Jitter solo se il glitch è attivo (con glitch = 0 nessuno spostamento)
                x_offset = y_offset = 0
                if (self.glitch_effect > 0 and intensity > 0.6
                        and rng.random() < self.glitch_effect):
                    x_offset = rng.randint(-jitter_x, jitter_x)
                    y_offset = rng.randint(-jitter_y, jitter_y)

                x_start = max(0, i * block_width + x_offset)
                y_start = max(0, j * block_height + y_offset)
                x_end = min(width, x_start + actual_width)
                y_end = min(height, y_start + actual_height)

                color_idx = int((intensity + i / num_blocks_x + j / num_blocks_y)
                                * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                brightness = 0.4 + intensity * 0.6 * self.master_intensity
                final_color = [c * brightness for c in color]

                if x_start < x_end and y_start < y_end:
                    pattern[y_start:y_end, x_start:x_end] = final_color

        return pattern

    # ── pattern 2 ────────────────────────────────────────────────────────────
    def pattern_2_horizontal_stripes_glitch(self, audio_idx, width, height):
        rng = self._rng(audio_idx)
        px = self._px(width, height)
        pattern = np.full((height, width, 3), self.background_color, dtype=np.float32)
        freq_data = self._freq(audio_idx)

        stripe_height = max(1, int(round(4 * self.thickness * px)))
        max_random_h = max(2, int(round(8 * px)))
        jitter = max(1, int(round(20 * px)))
        y = 0
        stripe_idx = 0

        while y < height:
            # Livello effettivo 0..1: le soglie restano coerenti con qualsiasi master
            level = min(1.0, float(freq_data[stripe_idx % len(freq_data)]) * self.master_intensity)

            current_stripe_height = stripe_height
            if level > 0.8:
                current_stripe_height = rng.randint(1, max_random_h)

            if level > 0.5:
                stripe_width = width
                x_start = 0
            else:
                stripe_width = max(1, int(width * (0.3 + level * 0.7)))
                x_start = rng.randint(0, max(0, width - stripe_width))

            if level > 0.7 and rng.random() < self.glitch_effect:
                x_start = max(0, min(width - stripe_width,
                                     x_start + rng.randint(-jitter, jitter)))

            color_idx = int((stripe_idx * 0.1 + level) * len(self.colors)) % len(self.colors)
            base_color = list(self.colors[color_idx])

            if level > 0.6 and rng.random() < self.glitch_effect:
                base_color[rng.randint(0, 2)] = rng.random()

            brightness = 0.3 + level * 0.7
            final_color = [c * brightness for c in base_color]

            y_end = min(height, y + current_stripe_height)
            x_end = min(width, x_start + stripe_width)

            if x_start < x_end and y < y_end:
                pattern[y:y_end, x_start:x_end] = final_color

            y += current_stripe_height
            stripe_idx += 1

        return pattern

    # ── pattern 3 ────────────────────────────────────────────────────────────
    def pattern_3_curved_flowing_lines(self, audio_idx, width, height):
        """Linee curve fluide, vettorizzate."""
        px = self._px(width, height)
        pattern = np.full((height, width, 3), self.background_color, dtype=np.float32)
        freq_data = self._freq(audio_idx)

        num_curves = int(8 + float(np.mean(freq_data)) * 15)
        x_coords = np.arange(width)
        freq_scale = 1280.0 / width  # la lunghezza d'onda segue la larghezza del frame

        for curve_idx in range(num_curves):
            intensity = float(freq_data[curve_idx % len(freq_data)])

            if intensity < 0.2 * self.master_intensity:
                continue

            amplitude = height * 0.3 * intensity * self.master_intensity
            frequency = (curve_idx + 1) * 0.02 * freq_scale
            phase = audio_idx * 0.1 + curve_idx * 0.5 * self.glitch_effect
            center_y = height * (curve_idx / num_curves)

            curve_y = (center_y + amplitude * np.sin(x_coords * frequency + phase)).astype(int)

            line_thickness = max(1, int(round(5 * intensity * self.thickness * px)))
            half = line_thickness // 2  # simmetrico attorno alla curva
            color_variation = 0.8 + intensity * 0.4 * self.glitch_effect
            final_color = np.array([c * color_variation for c in self.colors[curve_idx % len(self.colors)]],
                                   dtype=np.float32)

            denom = half + 1.0
            for t in range(-half, half + 1):
                ys = np.clip(curve_y + t, 0, height - 1)
                alpha = float((1.0 - abs(t) / denom) * intensity)
                pattern[ys, x_coords, :] = np.maximum(pattern[ys, x_coords, :], final_color * alpha)

        # Leggero smoothing orizzontale su tutti i canali in una sola chiamata
        pattern = gaussian_filter1d(pattern, sigma=0.5, axis=1)
        return pattern


# ─── Audio ────────────────────────────────────────────────────────────────────
def detect_bit_depth(audio_path):
    """Profondità di bit reale del file, quando il formato la espone."""
    try:
        subtype = soundfile.info(audio_path).subtype
    except Exception:
        return "n/d (lossy)"
    labels = {
        "PCM_S8": "8-bit", "PCM_U8": "8-bit", "PCM_16": "16-bit", "PCM_24": "24-bit",
        "PCM_32": "32-bit", "FLOAT": "32-bit Float", "DOUBLE": "64-bit Float",
    }
    return labels.get(subtype, "n/d (lossy)")


def extract_audio_features(audio_file):
    """Estrae le caratteristiche audio — robusto contro errori di shape e dtype"""
    try:
        y, sr = librosa.load(audio_file, sr=None, mono=True)

        if len(y) == 0:
            st.error("Il file audio sembra essere vuoto o corrotto")
            return None

        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)

        n_freq_bins = min(50, magnitude.shape[0])
        bands = magnitude[:n_freq_bins, :]
        frame_max = bands.max(axis=0, keepdims=True)
        bands = bands / np.where(frame_max > 0, frame_max, 1.0)
        spectral_features = bands.T.astype(np.float32)  # (n_frames, n_freq_bins)

        # beat_track può restituire array in versioni recenti di librosa
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.atleast_1d(tempo_raw).flat[0])

        return {
            'spectral_features': spectral_features,
            'tempo': tempo_val,
            'beats': beats,
            'duration': float(len(y)) / float(sr),
            'sample_rate': int(sr),
            'peak': float(np.max(np.abs(y))),
            'bit_info': detect_bit_depth(audio_file),
            'audio_path': audio_file
        }

    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {str(e)}")
        st.info("Suggerimento: prova con un file WAV o MP3 più piccolo")
        return None


# ─── Video ────────────────────────────────────────────────────────────────────
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
    try:
        return ImageFont.load_default(size)  # Pillow recenti: font scalabile
    except TypeError:
        return ImageFont.load_default()


def fit_title_font(text, width, height, padding):
    """Font del titolo scalato sulla risoluzione e ridotto finché il testo sta nel frame."""
    size = max(12, int(60 * min(width, height) / 720))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while size > 12:
        font = find_font(size)
        bbox = probe.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= width - 2 * padding:
            return font
        size -= 4
    return find_font(12)


def get_ffmpeg_exe():
    """Preferisce il binario di imageio-ffmpeg (già richiesto per scrivere il video)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


def create_video_with_audio(audio_path, video_path_no_audio, final_video_path):
    """Combina video e audio con FFmpeg."""
    ffmpeg = get_ffmpeg_exe()
    try:
        subprocess.run([ffmpeg, '-version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        st.error("FFmpeg non trovato. Installa FFmpeg (o imageio-ffmpeg) per combinare audio e video.")
        return False

    try:
        command = [
            ffmpeg, '-y',
            '-i', video_path_no_audio,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-shortest',
            final_video_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Errore nella combinazione di video e audio: {e.stderr}")
        return False


def generate_preview_frame(audio_features, pattern_type, user_params, position=0.5,
                           width=320, height=180):
    """Frame di anteprima a bassa risoluzione nella posizione indicata (0..1)."""
    generator = PatternGenerator(audio_features, user_params)
    n = len(audio_features['spectral_features'])
    idx = int(np.clip(position, 0.0, 1.0) * (n - 1))
    pattern = generator.render_frame(pattern_type, idx, width, height)
    frame_rgb = np.clip(pattern * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_rgb)


def render_video(audio_features, pattern_type, user_params, width, height, fps,
                 video_title="", title_pos_v=None, title_pos_h=None, progress_cb=None):
    """Renderizza il video completo e restituisce i bytes dell'MP4 (o None in caso di errore)."""
    work_dir = tempfile.mkdtemp(prefix="avpg_render_")
    writer = None
    try:
        video_no_audio_path = os.path.join(work_dir, "video_no_audio.mp4")
        final_video_path = os.path.join(work_dir, "final_video.mp4")

        generator = PatternGenerator(audio_features, user_params)
        total_frames = max(1, int(audio_features['duration'] * fps))
        audio_frame_step = len(audio_features['spectral_features']) / total_frames

        # Titolo: posizione e font calcolati una volta sola
        title_xy = title_font = None
        if video_title:
            padding = max(10, int(40 * min(width, height) / 720))
            title_font = fit_title_font(video_title, width, height, padding)
            probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            bbox = probe.textbbox((0, 0), video_title, font=title_font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if title_pos_h == "A Sinistra":
                x = padding
            elif title_pos_h == "A Destra":
                x = width - text_w - padding
            else:
                x = (width - text_w) / 2
            y = padding if title_pos_v == "In Alto" else height - text_h - padding
            title_xy = (x - bbox[0], y - bbox[1])

        writer = imageio.get_writer(video_no_audio_path, fps=fps, codec='libx264',
                                    quality=8, macro_block_size=1)

        for frame_idx in range(total_frames):
            audio_idx = int(frame_idx * audio_frame_step)
            pattern = generator.render_frame(pattern_type, audio_idx, width, height)
            frame_rgb = np.clip(pattern * 255, 0, 255).astype(np.uint8)

            if title_xy is not None:
                pil_img = Image.fromarray(frame_rgb)
                ImageDraw.Draw(pil_img).text(title_xy, video_title, font=title_font,
                                             fill=(255, 255, 255), stroke_width=2,
                                             stroke_fill=(0, 0, 0))
                frame_rgb = np.array(pil_img)

            writer.append_data(frame_rgb)

            if progress_cb and (frame_idx % 5 == 0 or frame_idx == total_frames - 1):
                progress_cb((frame_idx + 1) / total_frames, "Creazione video")

        writer.close()
        writer = None

        if progress_cb:
            progress_cb(1.0, "Combinando video e audio...")
        if not create_video_with_audio(audio_features['audio_path'],
                                       video_no_audio_path, final_video_path):
            return None
        if not os.path.exists(final_video_path):
            return None
        with open(final_video_path, 'rb') as vf:
            return vf.read()
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


# ─── Report ───────────────────────────────────────────────────────────────────
def generate_report_text(audio_features, pattern_type, aspect_ratio, user_params,
                         video_title="", filename=""):
    """Genera il report stilizzato da copiare/incollare per social/YouTube"""

    duration = audio_features['duration']
    tempo = audio_features['tempo']
    sr = audio_features['sample_rate']
    mins = int(duration // 60)
    secs = int(duration % 60)
    sr_khz = f"{sr / 1000:g}"
    bit_info = audio_features.get('bit_info', 'n/d')
    seed = int(user_params.get('seed', 0))

    # STYLE e PROCESS descrivono l'app, non il pattern
    style = "Spectral Pattern Synthesis / Real-Time Frequency Mapping"
    process = "STFT Analysis → Spectral Normalization → Frame Synthesis → H.264 Encode"
    quote_en = "It's not a video. It's the music rendered as light."
    quote_it = "Non è un video. È la musica resa come luce."

    fn_clean = filename.replace(".mp4", "").upper() if filename else pattern_type.upper().replace(" ", "_")

    # Clipping basato sul picco reale del segnale
    peak = audio_features.get('peak', 0.0)
    if peak >= 0.99:
        clip_en, clip_it = "Clipping_Point", "Punto_di_Clipping"
    elif peak >= 0.7:
        clip_en, clip_it = "Near_Clipping", "Vicino_al_Clipping"
    else:
        clip_en, clip_it = "Sub_Clipping", "Sotto_Clipping"

    title_en = f":: TITLE: {video_title}\n" if video_title else ""
    title_it = f":: TITOLO: {video_title}\n" if video_title else ""

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
:: AUDIO: {sr_khz}kHz / {bit_info} / {tempo:.1f} BPM / {clip_en}
:: PATTERN: {pattern_type}
:: FORMAT: {aspect_ratio}
:: SEED: {seed}
{title_en}:: PROCESS: {process}
:: DURATION: {mins:02d}:{secs:02d}

"{quote_en}"
> Direction & Algorithm: Loop507

{all_tags}



[AVPG_ARCHIVE] // {fn_clean} // H.264 // AAC

:: STILE: {style}
:: MOTORE: audio_visual_pattern_generator [02.00]
:: AUDIO: {sr_khz} kHz / {bit_info} / {tempo:.1f} BPM / {clip_it}
:: PATTERN: {pattern_type}
:: FORMATO: {aspect_ratio}
:: SEED: {seed}
{title_it}:: PROCESSO: {process}
:: DURATA: {mins:02d}:{secs:02d}

"{quote_it}"
> Regia e Algoritmo: Loop507

{all_tags}"""

    return report.strip()


# ─── Utilità UI ───────────────────────────────────────────────────────────────
def show_image(img, caption=None):
    """st.image compatibile: use_container_width è deprecato in favore di width='stretch'."""
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


def reset_audio_state():
    """Elimina l'audio temporaneo corrente e i risultati collegati."""
    audio_dir = st.session_state.get('audio_dir')
    if audio_dir:
        shutil.rmtree(audio_dir, ignore_errors=True)
    st.session_state.audio_dir = None
    st.session_state.audio_key = None
    st.session_state.audio_features = None
    st.session_state.video_bytes = None
    st.session_state.video_filename = None
    st.session_state.report_text = None
    st.session_state.report_filename = None


# ─── Session State ────────────────────────────────────────────────────────────
if 'user_colors' not in st.session_state:
    st.session_state.user_colors = []
    st.session_state.num_colors = 4
for _key in ('video_bytes', 'video_filename', 'report_text', 'report_filename',
             'audio_dir', 'audio_key', 'audio_features'):
    if _key not in st.session_state:
        st.session_state[_key] = None

# ─── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Carica un file audio (MP3, WAV, M4A, FLAC)",
    type=['mp3', 'wav', 'm4a', 'flac']
)

if uploaded_file is None:
    if st.session_state.audio_key is not None:
        reset_audio_state()
else:
    # L'analisi audio si rifà solo quando cambia il file, non a ogni movimento di slider
    file_key = (uploaded_file.name, uploaded_file.size)
    if st.session_state.audio_key != file_key:
        reset_audio_state()
        try:
            audio_dir = tempfile.mkdtemp(prefix="avpg_audio_")
            temp_audio_path = os.path.join(audio_dir, os.path.basename(uploaded_file.name))
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_dir = audio_dir

            with st.spinner("Analizzando l'audio..."):
                features = extract_audio_features(temp_audio_path)

            if features is not None:
                st.session_state.audio_features = features
                st.session_state.audio_key = file_key
            else:
                reset_audio_state()
        except Exception as e:
            st.error(f"Errore nel caricamento del file: {str(e)}")
            reset_audio_state()

audio_features = st.session_state.audio_features

if uploaded_file is not None and audio_features is not None:
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
        pattern_type = st.selectbox("Seleziona il tipo di pattern:", PATTERN_NAMES)
    with col_slider:
        aspect_ratio = st.selectbox(
            "Seleziona l'aspect ratio:",
            ["1:1 (Square)", "9:16 (Verticale)", "16:9 (Orizzontale)"]
        )

    col_seed, col_fps, col_res = st.columns(3)
    with col_seed:
        seed = int(st.number_input("Seed", min_value=0, max_value=999999, value=507, step=1,
                                   help="Stesso seed + stessi parametri = stesso video."))
    with col_fps:
        fps = st.selectbox("FPS", [24, 30], index=1,
                           help="24 fps rende il render circa il 20% più veloce.")
    with col_res:
        render_quality = st.selectbox("Risoluzione render", ["Piena", "Metà (veloce)"])

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
        if custom_colors:
            num_colors = st.slider("Numero di colori", 2, 8,
                                   st.session_state.num_colors, key='num_colors_slider')
            st.session_state.num_colors = num_colors
            if len(st.session_state.user_colors) != num_colors:
                color_rng = random.Random(seed)
                st.session_state.user_colors = [
                    "#%06x" % color_rng.randint(0, 0xFFFFFF) for _ in range(num_colors)
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
            user_colors = None  # palette generata dal seed

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

    user_params = {
        "seed": seed,
        "master_intensity": master_intensity,
        "glitch_effect": glitch_effect,
        "thickness": thickness,
        "colors": user_colors,
        "background_color": bg_color_rgb
    }

    # ─── Anteprima rapida ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🖼️ Anteprima (bassa risoluzione)")
    preview_pos = st.slider("Posizione nel brano (%)", 0, 100, 50)

    with st.spinner("Generando anteprima..."):
        preview_img = generate_preview_frame(audio_features, pattern_type, user_params,
                                             position=preview_pos / 100.0)
    show_image(preview_img, caption=f"Pattern: {pattern_type} — seed {seed}")

    # ─── Generazione video ────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🎬 Genera Video MP4"):
        if aspect_ratio == "1:1 (Square)":
            width, height = 1080, 1080
        elif aspect_ratio == "9:16 (Verticale)":
            width, height = 720, 1280
        else:
            width, height = 1280, 720
        if render_quality != "Piena":
            width, height = width // 2, height // 2
        # Dimensioni pari, richieste da H.264 / yuv420p
        width -= width % 2
        height -= height % 2

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(value, label):
            progress_bar.progress(min(1.0, value))
            status_text.text(f"{label}: {int(value * 100)}%")

        try:
            with st.spinner("Generando video... Questo può richiedere alcuni minuti."):
                video_bytes = render_video(
                    audio_features, pattern_type, user_params, width, height, fps,
                    video_title=video_title, title_pos_v=title_position_v,
                    title_pos_h=title_position_h, progress_cb=on_progress
                )

            if video_bytes:
                video_filename = f"pattern_{pattern_type.replace(' ', '_')}_s{seed}.mp4"
                st.session_state.video_bytes = video_bytes
                st.session_state.video_filename = video_filename
                st.session_state.report_text = generate_report_text(
                    audio_features=audio_features,
                    pattern_type=pattern_type,
                    aspect_ratio=aspect_ratio,
                    user_params=user_params,
                    video_title=video_title,
                    filename=video_filename
                )
                st.session_state.report_filename = f"report_{pattern_type.replace(' ', '_')}_s{seed}.txt"
                st.success("✅ Video generato con successo!")
            else:
                st.error("Il file video finale non è stato creato.")
        except Exception as e:
            st.error(f"Errore durante la generazione: {str(e)}")
            st.info("Prova a ricaricare l'app e riprovare con un file diverso.")
        finally:
            progress_bar.empty()
            status_text.empty()

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
