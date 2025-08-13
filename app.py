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
        # Carica l'audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        if len(y) == 0:
            st.error("Il file audio sembra essere vuoto o corrotto")
            return None
            
        # Calcola lo spettrogramma
        stft = librosa.stft(y, hop_length=512)
        magnitude = np.abs(stft)
        
        # Features spettrali nel tempo
        spectral_features = []
        for frame in range(magnitude.shape[1]):
            # Prendi le prime 50 frequenze e normalizza
            frame_data = magnitude[:50, frame]
            if np.max(frame_data) > 0:
                frame_data = frame_data / np.max(frame_data)
            else:
                frame_data = np.zeros_like(frame_data)
            spectral_features.append(frame_data)
        
        # Altre features con gestione errori
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if tempo is not None else 120.0
        except:
            tempo = 120.0  # Valore di default
            beats = np.array([])
        
        return {
            'spectral_features': spectral_features,
            'tempo': tempo,
            'beats': beats,
            'duration': len(y) / sr,
            'sample_rate': sr
        }
    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {str(e)}")
        return None

def create_animated_visualization(audio_features, pattern_type):
    """Crea la visualizzazione animata"""
    generator = PatternGenerator(audio_features)
    
    # Configurazione dell'animazione
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Numero totale di frame basato sulla durata
    total_frames = len(audio_features['spectral_features'])
    
    # Funzione di aggiornamento per l'animazione
    def update_frame(frame_idx):
        ax.clear()
        ax.axis('off')
        
        # Genera il pattern basato sul tipo selezionato
        if pattern_type == "Blocchi Glitch":
            pattern = generator.pattern_1_glitch_blocks(frame_idx)
        elif pattern_type == "Strisce Orizzontali":
            pattern = generator.pattern_2_horizontal_stripes_glitch(frame_idx)
        else:  # Linee Curve Fluide
            pattern = generator.pattern_3_curved_flowing_lines(frame_idx)
        
        ax.imshow(pattern, aspect='auto')
        ax.set_title(f"Frame {frame_idx + 1}/{total_frames}", color='white', fontsize=12)
        
        return [ax]
    
    return fig, update_frame, total_frames

# Interface Streamlit
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
            
            **Caratteristiche:**
            - Colori casuali generati ad ogni esecuzione
            - Sincronizzazione con le frequenze audio
            - Effetti di glitch e distorsione
            - Pattern mai identici tra esecuzioni diverse
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
    """)
    
    st.markdown("### 🚀 Deploy su Streamlit")
    st.markdown("""
    Per deployare su Streamlit Cloud:
    1. Carica il codice su GitHub
    2. Aggiungi requirements.txt
    3. Connetti il repo a Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("💡 *I pattern sono generati in tempo reale e sono sempre unici!*")
