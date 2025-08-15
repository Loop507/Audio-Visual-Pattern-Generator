import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import io
import random
import time
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import colorsys
import imageio
from PIL import Image
import os
import tempfile

# Prova a importare OpenCV, se fallisce usa solo imageio
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV non disponibile, usando solo imageio per i video")

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
    
    def create_colormap(self):
        """Crea una colormap personalizzata con i colori generati"""
        return LinearSegmentedColormap.from_list("custom", self.colors, N=256)
    
    def pattern_1_glitch_blocks(self, frame_idx, width=400, height=300):
        """Pattern 1: Blocchi colorati glitch come nella prima immagine"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        # Crea blocchi di diverse dimensioni con effetto glitch
        num_blocks_x = random.randint(15, 40)
        num_blocks_y = random.randint(10, 25)
        
        block_width = width // num_blocks_x
        block_height = height // num_blocks_y
        
        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                # Intensità basata sulla posizione e frequenza
                freq_idx = (i + j) % len(freq_data)
                intensity = freq_data[freq_idx]
                
                # Skip alcuni blocchi per effetto glitch
                if intensity < 0.3 and random.random() < 0.4:
                    continue
                
                # Dimensioni variabili del blocco
                actual_width = int(block_width * (0.5 + intensity * 0.5))
                actual_height = int(block_height * (0.5 + intensity * 0.5))
                
                # Posizione con offset casuale per glitch
                x_offset = random.randint(-5, 5) if intensity > 0.7 else 0
                y_offset = random.randint(-3, 3) if intensity > 0.6 else 0
                
                x_start = max(0, i * block_width + x_offset)
                y_start = max(0, j * block_height + y_offset)
                x_end = min(width, x_start + actual_width)
                y_end = min(height, y_start + actual_height)
                
                # Colore basato su intensità e posizione
                color_idx = int((intensity + i/num_blocks_x + j/num_blocks_y) * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                
                # Variazione di luminosità
                brightness = 0.4 + intensity * 0.6
                final_color = [c * brightness for c in color]
                
                if x_start < x_end and y_start < y_end:
                    pattern[y_start:y_end, x_start:x_end] = final_color
                    
        return pattern
    
    def pattern_2_horizontal_stripes_glitch(self, frame_idx, width=400, height=300):
        """Pattern 2: Strisce orizzontali con glitch digitale come nella seconda immagine"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        # Strisce orizzontali sottili
        stripe_height = random.randint(1, 4)  # Strisce molto sottili
        
        y = 0
        stripe_idx = 0
        
        while y < height:
            # Intensità per questa striscia
            freq_idx = stripe_idx % len(freq_data)
            intensity = freq_data[freq_idx]
            
            # Altezza variabile della striscia
            current_stripe_height = stripe_height
            if intensity > 0.8:  # Glitch per alta intensità
                current_stripe_height = random.randint(1, 8)
            
            # Larghezza della striscia (può non coprire tutto)
            if intensity > 0.5:
                stripe_width = width  # Striscia completa
                x_start = 0
            else:
                stripe_width = int(width * (0.3 + intensity * 0.7))
                x_start = random.randint(0, max(1, width - stripe_width))
            
            # Glitch orizzontale - spostamento casuale
            if intensity > 0.7 and random.random() < 0.3:
                x_offset = random.randint(-20, 20)
                x_start = max(0, min(width - stripe_width, x_start + x_offset))
            
            # Colore per questa striscia
            color_idx = int((stripe_idx * 0.1 + intensity) * len(self.colors)) % len(self.colors)
            base_color = self.colors[color_idx]
            
            # Variazione di colore per effetto glitch
            color = list(base_color)
            if intensity > 0.6 and random.random() < 0.2:
                # Glitch di colore
                color[random.randint(0, 2)] = random.random()
            
            # Modula luminosità
            brightness = 0.3 + intensity * 0.7
            final_color = [c * brightness for c in color]
            
            # Disegna la striscia
            y_end = min(height, y + current_stripe_height)
            x_end = min(width, x_start + stripe_width)
            
            if x_start < x_end and y < y_end:
                pattern[y:y_end, x_start:x_end] = final_color
            
            y += current_stripe_height
            stripe_idx += 1
            
        return pattern
    
    def pattern_3_curved_flowing_lines(self, frame_idx, width=400, height=300):
        """Pattern 3: Linee curve fluide come nella terza immagine"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        # Numero di linee curve basato sulle frequenze
        num_curves = int(8 + np.mean(freq_data) * 15)
        
        # Crea griglia per le coordinate
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        for curve_idx in range(num_curves):
            freq_idx = curve_idx % len(freq_data)
            intensity = freq_data[freq_idx]
            
            if intensity < 0.2:  # Skip curve con bassa intensità
                continue
            
            # Parametri per la curva
            amplitude = height * 0.3 * intensity  # Ampiezza dell'onda
            frequency = (curve_idx + 1) * 0.02  # Frequenza dell'onda
            phase = frame_idx * 0.1 + curve_idx * 0.5  # Fase per animazione
            
            # Centro verticale della curva
            center_y = height * (curve_idx / num_curves)
            
            # Calcola la curva sinusoidale
            curve_y = center_y + amplitude * np.sin(x_coords * frequency + phase)
            
            # Spessore della linea basato sull'intensità
            line_thickness = max(1, int(5 * intensity))
            
            # Colore per questa curva
            color_idx = curve_idx % len(self.colors)
            base_color = self.colors[color_idx]
            
            # Modifica colore per varietà
            color_variation = 0.8 + intensity * 0.4
            final_color = [c * color_variation for c in base_color]
            
            # Disegna la curva
            for x in range(width):
                curve_center = int(curve_y[0, x])
                
                # Disegna lo spessore della linea
                for thickness in range(-line_thickness//2, line_thickness//2 + 1):
                    y_pos = curve_center + thickness
                    
                    if 0 <= y_pos < height:
                        # Effetto sfumato per lo spessore
                        alpha = 1.0 - abs(thickness) / (line_thickness/2 + 1)
                        alpha *= intensity  # Modula con intensità audio
                        
                        # Mescola il colore
                        for c in range(3):
                            pattern[y_pos, x, c] = max(pattern[y_pos, x, c], 
                                                     final_color[c] * alpha)
        
        # Effetto di flow - aggiungi movimento fluido
        if frame_idx > 0:
            # Leggero blur orizzontale per effetto flow
            for c in range(3):
                pattern[:, :, c] = gaussian_filter1d(pattern[:, :, c], 
                                                   sigma=0.5, axis=1)
        
        return pattern

def extract_audio_features(audio_file):
    """Estrae le caratteristiche audio per la visualizzazione"""
    try:
        # Carica l'audio con parametri più compatibili
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        
        if len(y) == 0:
            st.error("Il file audio sembra essere vuoto o corrotto")
            return None
            
        # Calcola lo spettrogramma con parametri fissi
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Limita il numero di features per performance
        n_freq_bins = min(50, magnitude.shape[0])
        
        # Features spettrali nel tempo
        spectral_features = []
        step = max(1, magnitude.shape[1] // 1000)  # Massimo 1000 frame
        
        for frame in range(0, magnitude.shape[1], step):
            # Prendi le prime n_freq_bins frequenze e normalizza
            frame_data = magnitude[:n_freq_bins, frame]
            if np.max(frame_data) > 0:
                frame_data = frame_data / np.max(frame_data)
            else:
                frame_data = np.zeros(n_freq_bins)
            spectral_features.append(frame_data)
        
        # Beat tracking più semplice
        try:
            # Usa onset detection più semplice se beat tracking fallisce
            tempo = None
            beats = None
            
            # Prova prima con onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                # Stima il tempo dagli onset
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
            
        except Exception as beat_error:
            st.warning(f"Impossibile estrarre il beat: {beat_error}")
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
        st.info("Suggerimento: prova con un file WAV o MP3 più piccolo")
        return None

def create_video_from_patterns(audio_features, pattern_type, fps=30, max_duration=30):
    """
    Genera un video MP4 a partire dai pattern visivi - VERSIONE CORRETTA
    
    :param audio_features: Dizionario con le caratteristiche audio.
    :param pattern_type: Tipo di pattern da generare.
    :param fps: Frame per secondo del video.
    :param max_duration: Durata massima del video in secondi.
    :return: Percorso del file video creato.
    """
    try:
        # Limita la durata per evitare file troppo grandi
        video_duration = min(audio_features['duration'], max_duration)
        num_frames = int(video_duration * fps)
        
        # Risoluzione fissa per evitare problemi
        width, height = 800, 600
        
        # Crea il generatore di pattern
        generator = PatternGenerator(audio_features)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Lista per contenere i frame
        frames = []
        
        # Genera tutti i frame
        for i in range(num_frames):
            # Aggiorna progress
            progress = i / num_frames
            progress_bar.progress(progress)
            status_text.text(f"Generando frame {i+1}/{num_frames}")
            
            # Calcola l'indice del frame audio da usare
            audio_frame_idx = int(i * len(audio_features['spectral_features']) / num_frames)
            
            # Seleziona e genera il pattern
            if pattern_type == "Blocchi Glitch":
                pattern = generator.pattern_1_glitch_blocks(audio_frame_idx, width, height)
            elif pattern_type == "Strisce Orizzontali":
                pattern = generator.pattern_2_horizontal_stripes_glitch(audio_frame_idx, width, height)
            else:  # Linee Curve Fluide
                pattern = generator.pattern_3_curved_flowing_lines(audio_frame_idx, width, height)
            
            # Assicurati che i valori siano nel range corretto [0, 1]
            pattern = np.clip(pattern, 0, 1)
            
            # Converte in uint8 per il video
            frame_uint8 = (pattern * 255).astype(np.uint8)
            frames.append(frame_uint8)
        
        # Pulisci la progress bar
        progress_bar.empty()
        status_text.text("Salvando il video...")
        
        # Crea il file temporaneo
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = temp_video.name
        temp_video.close()
        
        # Scrivi il video usando imageio
        with imageio.get_writer(video_path, fps=fps, format='mp4', codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)
        
        status_text.text("Video completato!")
        
        # Verifica che il file sia stato creato correttamente
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            return video_path
        else:
            raise Exception("Il file video non è stato creato correttamente")
            
    except Exception as e:
        st.error(f"Errore nella generazione video: {str(e)}")
        # Cleanup in caso di errore
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        return None

# --- Interfaccia Streamlit ---
uploaded_file = st.file_uploader(
    "Carica un file audio (MP3, WAV, M4A)", 
    type=['mp3', 'wav', 'm4a', 'flac']
)

if uploaded_file is not None:
    try:
        # Salva il file temporaneamente
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File audio caricato con successo!")
        
        # Estrai le features audio
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
                
                # Crea il generatore di pattern
                generator = PatternGenerator(audio_features)
                
                # Mostra alcuni frame statici come anteprima
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
                
                # SEZIONE GENERAZIONE VIDEO CORRETTA
                st.subheader("🎬 Genera Video Animato")
                
                st.info("💡 Il video sarà limitato a massimo 30 secondi per performance ottimali")
                
                # Opzioni per il video
                col_fps, col_duration = st.columns(2)
                with col_fps:
                    video_fps = st.selectbox("FPS Video", [24, 30], index=1)
                with col_duration:
                    max_duration = st.slider("Durata massima (secondi)", 5, 30, 15)
                
                if st.button("🎬 Genera Video MP4", key="video_btn"):
                    try:
                        video_path = create_video_from_patterns(
                            audio_features, 
                            pattern_type, 
                            fps=video_fps,
                            max_duration=max_duration
                        )
                        
                        if video_path and os.path.exists(video_path):
                            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                            st.success(f"✅ Video generato! Dimensione: {file_size:.2f} MB")
                            
                            # Leggi il file video
                            with open(video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            # Mostra il video
                            st.video(video_bytes)
                            
                            # Pulsante di download
                            st.download_button(
                                label="📥 Scarica Video MP4",
                                data=video_bytes,
                                file_name=f"pattern_{pattern_type.replace(' ', '_')}_{int(time.time())}.mp4",
                                mime="video/mp4"
                            )
                            
                            # Pulisci il file temporaneo
                            try:
                                os.remove(video_path)
                            except:
                                pass
                        else:
                            st.error("❌ Errore nella generazione del video")
                            
                    except Exception as e:
                        st.error(f"❌ Errore: {str(e)}")
                        st.info("💡 Prova con un FPS più basso o una durata minore")
                
                # Informazioni sui colori
                st.subheader("Palette Colori Generata")
                color_cols = st.columns(len(generator.colors))
                for i, (col, color) in enumerate(zip(color_cols, generator.colors)):
                    with col:
                        # Crea un quadrato colorato
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
            **Pattern Generati:**
            
            1. **Blocchi Glitch**: Blocchi colorati di diverse dimensioni con effetti di glitch digitale
            2. **Strisce Orizzontali**: Linee orizzontali sottili con distorsioni e spostamenti
            3. **Linee Curve Fluide**: Curve sinusoidali che scorrono fluidamente seguendo l'audio
            
            **Caratteristiche Video:**
            - Risoluzione: 800x600 pixel
            - Durata massima: 30 secondi
            - Formato: MP4 (codec H.264)
            - Sincronizzazione con frequenze audio
            - Pattern sempre unici
            
            **Problemi Risolti:**
            - Gestione corretta dei frame
            - Formato colori compatibile
            - Gestione memoria ottimizzata
            - Progress bar per il feedback
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
    - **Esportazione Video MP4** ✅
    """)
    
    st.markdown("### 🔧 Requirements.txt")
    st.code("""streamlit
numpy
librosa
matplotlib
scipy
pillow
imageio[ffmpeg]
colorsys""", language="text")

# Footer
st.markdown("---")
st.markdown("💡 *I pattern sono generati in tempo reale e sono sempre unici!*")
