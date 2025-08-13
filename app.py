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
    
    def pattern_1_vertical_bars(self, frame_idx, width=400, height=300):
        """Pattern 1: Barre verticali colorate con glitch effect"""
        pattern = np.zeros((height, width, 3))
        
        # Usa le frequenze per modulare le barre
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        # Numero di barre basato sulle frequenze
        num_bars = int(20 + np.mean(freq_data) * 50)
        bar_width = width // num_bars
        
        for i in range(num_bars):
            # Intensità basata sulle frequenze
            intensity = freq_data[i % len(freq_data)]
            
            # Altezza della barra modulata dalle frequenze
            bar_height = int(height * (0.3 + intensity * 0.7))
            start_y = (height - bar_height) // 2
            
            # Colore basato sulla posizione e frequenza
            color_idx = int((i / num_bars + intensity) * len(self.colors)) % len(self.colors)
            color = self.colors[color_idx]
            
            # Glitch effect
            if random.random() < 0.1:  # 10% chance di glitch
                start_y += random.randint(-20, 20)
                bar_height = min(height - start_y, bar_height)
            
            # Disegna la barra
            x_start = i * bar_width
            x_end = min((i + 1) * bar_width, width)
            y_end = min(start_y + bar_height, height)
            
            pattern[start_y:y_end, x_start:x_end] = color
            
        return pattern
    
    def pattern_2_horizontal_spectrum(self, frame_idx, width=400, height=300):
        """Pattern 2: Spettro orizzontale con effetti di distorsione"""
        pattern = np.zeros((height, width, 3))
        
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        # Crea linee orizzontali basate sulle frequenze
        for y in range(0, height, 2):  # Ogni 2 pixel
            freq_idx = int((y / height) * len(freq_data))
            intensity = freq_data[freq_idx]
            
            # Larghezza delle linee basata sull'intensità
            line_width = int(width * intensity)
            
            # Effetto di distorsione
            distortion = int(np.sin(frame_idx * 0.1 + y * 0.05) * 20)
            line_width = max(0, min(width, line_width + distortion))
            
            if line_width > 0:
                # Colore basato sulla frequenza e posizione
                color_idx = int((y / height + intensity) * len(self.colors)) % len(self.colors)
                color = self.colors[color_idx]
                
                # Gradiente orizzontale
                for x in range(line_width):
                    alpha = (x / line_width) if line_width > 0 else 0
                    pattern[y, x] = [c * alpha for c in color]
                    
        return pattern
    
    def pattern_3_circular_waves(self, frame_idx, width=400, height=300):
        """Pattern 3: Onde circolari con effetti radiali"""
        pattern = np.zeros((height, width, 3))
        
        center_x, center_y = width // 2, height // 2
        freq_data = self.audio_features['spectral_features'][frame_idx % len(self.audio_features['spectral_features'])]
        
        # Griglia di coordinate
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Distanza dal centro
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Angoli
        angles = np.arctan2(y_coords - center_y, x_coords - center_x)
        
        # Numero di onde basato sulle frequenze
        num_waves = int(5 + np.mean(freq_data) * 10)
        
        for i in range(num_waves):
            freq_idx = i % len(freq_data)
            intensity = freq_data[freq_idx]
            
            # Onda radiale
            wave_radius = 50 + i * 30
            wave = np.sin((distances - wave_radius) * 0.1 + frame_idx * 0.2) * intensity
            
            # Onda angolare
            angular_wave = np.sin(angles * (i + 1) + frame_idx * 0.1) * intensity
            
            # Combina le onde
            combined_wave = wave * angular_wave
            combined_wave = np.clip(combined_wave, 0, 1)
            
            # Colore per questa onda
            color = self.colors[i % len(self.colors)]
            
            # Applica il colore dove l'onda è positiva
            mask = combined_wave > 0.1
            for c in range(3):
                pattern[mask, c] = np.maximum(pattern[mask, c], 
                                            combined_wave[mask] * color[c])
                
        return pattern

def extract_audio_features(audio_file):
    """Estrae le caratteristiche audio per la visualizzazione"""
    try:
        # Carica l'audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Calcola lo spettrogramma
        stft = librosa.stft(y, hop_length=512)
        magnitude = np.abs(stft)
        
        # Features spettrali nel tempo
        spectral_features = []
        for frame in range(magnitude.shape[1]):
            # Prendi le prime 50 frequenze e normalizza
            frame_data = magnitude[:50, frame]
            frame_data = frame_data / (np.max(frame_data) + 1e-6)
            spectral_features.append(frame_data)
        
        # Altre features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            'spectral_features': spectral_features,
            'tempo': tempo,
            'beats': beats,
            'duration': len(y) / sr,
            'sample_rate': sr
        }
    except Exception as e:
        st.error(f"Errore nell'estrazione delle features audio: {e}")
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
        if pattern_type == "Barre Verticali":
            pattern = generator.pattern_1_vertical_bars(frame_idx)
        elif pattern_type == "Spettro Orizzontale":
            pattern = generator.pattern_2_horizontal_spectrum(frame_idx)
        else:  # Onde Circolari
            pattern = generator.pattern_3_circular_waves(frame_idx)
        
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
    # Salva il file temporaneamente
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File audio caricato con successo!")
    
    # Estrai le features audio
    with st.spinner("Analizzando l'audio..."):
        audio_features = extract_audio_features("temp_audio.wav")
    
    if audio_features is not None:
        st.success("Analisi audio completata!")
        
        # Mostra informazioni sull'audio
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Durata", f"{audio_features['duration']:.2f}s")
        with col2:
            st.metric("Tempo", f"{audio_features['tempo']:.1f} BPM")
        with col3:
            st.metric("Sample Rate", f"{audio_features['sample_rate']} Hz")
        
        # Selezione del tipo di pattern
        pattern_type = st.selectbox(
            "Seleziona il tipo di pattern:",
            ["Barre Verticali", "Spettro Orizzontale", "Onde Circolari"]
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
                        
                        if pattern_type == "Barre Verticali":
                            pattern = generator.pattern_1_vertical_bars(frame_idx)
                        elif pattern_type == "Spettro Orizzontale":
                            pattern = generator.pattern_2_horizontal_spectrum(frame_idx)
                        else:
                            pattern = generator.pattern_3_circular_waves(frame_idx)
                        
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
                if pattern_type == "Barre Verticali":
                    pattern = generator.pattern_1_vertical_bars(frame_slider)
                elif pattern_type == "Spettro Orizzontale":
                    pattern = generator.pattern_2_horizontal_spectrum(frame_slider)
                else:
                    pattern = generator.pattern_3_circular_waves(frame_slider)
                
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
            
            1. **Barre Verticali**: Le barre cambiano altezza e colore basandosi sulle frequenze audio
            2. **Spettro Orizzontale**: Linee orizzontali che rappresentano lo spettro delle frequenze
            3. **Onde Circolari**: Pattern radiali che si espandono seguendo il ritmo
            
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
