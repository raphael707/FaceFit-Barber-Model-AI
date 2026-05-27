import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceFit Barber — Data Dashboard",
    page_icon="🧔",
    layout="wide"
)

# ─── Warna & Konstanta ────────────────────────────────────────────────────────
CLASS_COLORS = {
    'oval'   : '#4CAF50',
    'round'  : '#2196F3',
    'square' : '#FF9800',
    'heart'  : '#E91E63',
    'diamond': '#9C27B0'
}
CLASSES = ['oval', 'round', 'square', 'heart', 'diamond']

warna_merah = '#E63946'
warna_hijau = '#2A9D8F'
warna_oren  = '#F4A261'
warna_biru  = '#457B9D'

# ─── Data Hardcoded dari Metadata W2 & W3 ────────────────────────────────────

# Dataset Overview
datasets_info = pd.DataFrame([
    {'Dataset': 'UTKFace',          'Total Raw': 66966, 'Masuk Pipeline': 9974,  'Filter': 'Asian only (race=2)',     'Label': 'MediaPipe'},
    {'Dataset': 'FairFace',         'Total Raw': 97698, 'Masuk Pipeline': 26047, 'Filter': 'East/Southeast Asian',   'Label': 'MediaPipe'},
    {'Dataset': 'Men Face Shape',   'Total Raw': 1309,  'Masuk Pipeline': 1004,  'Filter': 'Train split saja',       'Label': 'Sudah ada'},
    {'Dataset': 'Dataset ZIP',      'Total Raw': 1433,  'Masuk Pipeline': 1433,  'Filter': 'Hapus rect & triangle',  'Label': 'Sudah ada (5 kelas)'},
    {'Dataset': 'Indo Public Figure','Total Raw': 640,  'Masuk Pipeline': 640,   'Filter': '-',                      'Label': 'MediaPipe'},
])

# Hasil Labeling per Sumber
labeling_data = {
    'UTKFace'         : {'oval': 1160, 'round': 4,   'square': 6,   'heart': 68,  'diamond': 0},
    'FairFace'        : {'oval': 3541, 'round': 33,  'square': 20,  'heart': 219, 'diamond': 0},
    'Indo Public'     : {'oval': 436,  'round': 0,   'square': 0,   'heart': 0,   'diamond': 0},
    'Men Face Shape'  : {'oval': 337,  'round': 338, 'square': 312, 'heart': 0,   'diamond': 0},
    'Dataset ZIP'     : {'oval': 356,  'round': 338, 'square': 338, 'heart': 167, 'diamond': 178},
}

# Tahapan Balancing
stages = {
    'Sebelum\nAugmentasi': {'oval': 4603, 'round': 710,  'square': 671,  'heart': 413,  'diamond': 178},
    'Setelah\nAugmentasi': {'oval': 4603, 'round': 1100, 'square': 1100, 'heart': 1100, 'diamond': 1100},
    'Setelah\nValidasi'  : {'oval': 4601, 'round': 1097, 'square': 1097, 'heart': 1094, 'diamond': 1032},
    'Setelah\nBalancing' : {'oval': 1500, 'round': 1097, 'square': 1097, 'heart': 1094, 'diamond': 1032},
}

# Dataset Final
final_counts = {'oval': 1500, 'round': 1097, 'square': 1097, 'heart': 1094, 'diamond': 1032}
total_final  = sum(final_counts.values())
ir_final     = max(final_counts.values()) / min(final_counts.values())

# Survey
survey_freq      = {'Jarang': 10, 'Sebulan sekali': 9, '2-3 bulan sekali': 6}
survey_disap     = {'Tidak pernah': 3, 'Jarang': 6, 'Beberapa kali': 11, 'Sering': 5}
survey_causes    = {'Susah jelaskan\nke kapster': 9, 'Gaya tidak cocok\nbentuk wajah': 8, 'Referensi ≠\nhasil asli': 4, 'Lainnya': 4}
interest_scores  = [1, 2, 3, 4, 5]
interest_counts  = [5, 1, 2, 2, 15]
frek_kesulitan   = [4, 4, 3]
minat_app_kapster= [5, 5, 4]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧔 FaceFit Barber")
    st.caption("Data Science Dashboard — CC26-PSU304")
    st.divider()

    halaman = st.radio(
        "Navigasi",
        ["📊 Overview Dataset", "📋 Hasil Survey", "⚙️ Pipeline & Balancing", "✅ Dataset Final"]
    )
    st.divider()
    st.markdown("""
    **Tim:** CC26-PSU304  
    **Path:** Data Scientist  
    **Dikerjakan oleh:**  
    Diwan Ramadhani (W1, W2)  
    Rofiqho Izmi (W3 EDA)
    """)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("FaceFit Barber — Dataset Dashboard")
st.caption("Analisis data pipeline W1–W3: dari raw dataset ke dataset siap training model CNN")
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — OVERVIEW DATASET
# ═══════════════════════════════════════════════════════════════════════════════
if halaman == "📊 Overview Dataset":

    st.subheader("Overview 5 Dataset yang Digunakan")

    # Metrik
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dataset", "5 dataset")
    col2.metric("Total Raw (semua sumber)", f"{datasets_info['Total Raw'].sum():,}")
    col3.metric("Total Masuk Pipeline", f"{datasets_info['Masuk Pipeline'].sum():,}")
    col4.metric("Kelas Target", "5 kelas")
    st.markdown("")

    # Tabel dataset
    st.markdown("### Ringkasan Dataset")
    st.dataframe(datasets_info, use_container_width=True, hide_index=True)
    st.divider()

    # Bar chart: raw vs masuk pipeline
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Volume Raw vs Masuk Pipeline")
        fig, ax = plt.subplots(figsize=(9, 4))
        x  = np.arange(len(datasets_info))
        w  = 0.35
        b1 = ax.bar(x - w/2, datasets_info['Total Raw'],       color=warna_biru,  alpha=0.85, label='Total Raw')
        b2 = ax.bar(x + w/2, datasets_info['Masuk Pipeline'],  color=warna_hijau, alpha=0.85, label='Masuk Pipeline')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_info['Dataset'], rotation=10, ha='right', fontsize=9)
        ax.set_ylabel('Jumlah Gambar')
        ax.set_title('Perbandingan Volume: Raw vs Masuk Pipeline')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Anotasi
        for bar in b1:
            h = bar.get_height()
            if h > 5000:
                ax.text(bar.get_x() + bar.get_width()/2, h + 800, f'{h:,}', ha='center', fontsize=7.5)
        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 400, f'{h:,}', ha='center', fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### Distribusi Masuk Pipeline")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors_pie = [warna_merah, warna_hijau, warna_oren, warna_biru, '#9C27B0']
        explode    = [0.05] * 5
        ax.pie(
            datasets_info['Masuk Pipeline'],
            labels=datasets_info['Dataset'],
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        ax.set_title('Proporsi Kontribusi\nper Dataset')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Hasil Labeling
    st.markdown("### Hasil Labeling MediaPipe per Sumber")
    st.caption("Diamond = 0 di UTKFace, FairFace, dan Indo Public Figure — hanya ada di Dataset ZIP")

    col_la, col_lb = st.columns([3, 2])

    with col_la:
        df_label = pd.DataFrame(labeling_data).T
        fig, ax = plt.subplots(figsize=(10, 5))
        x  = np.arange(len(labeling_data))
        w  = 0.15
        for i, cls in enumerate(CLASSES):
            vals = [labeling_data[src][cls] for src in labeling_data]
            ax.bar(x + i * w - 2 * w, vals, w,
                   label=cls, color=CLASS_COLORS[cls], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(list(labeling_data.keys()), fontsize=9)
        ax.set_title('Distribusi Label per Sumber Dataset')
        ax.set_ylabel('Jumlah Gambar')
        ax.legend(title='Kelas', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_lb:
        st.markdown("**Insight Kunci:**")
        st.info("💡 **Diamond = 0** di UTKFace, FairFace, dan Indo Public. Wajah diamond langka di dataset *in-the-wild* Asia — threshold geometris MediaPipe sangat jarang terpenuhi.")
        st.warning("⚠️ **Indo Public 100% Oval** — foto publik figur Indonesia sering diambil dari sudut 3/4 sehingga rasio height/width masuk threshold oval.")
        st.success("✅ **Men Face Shape + Dataset ZIP** menjadi sumber utama untuk kelas round, square, heart, dan diamond.")

# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — HASIL SURVEY
# ═══════════════════════════════════════════════════════════════════════════════
elif halaman == "📋 Hasil Survey":

    st.subheader("Hasil Survey — 28 Responden (25 Pelanggan + 3 Kapster)")
    st.caption("Survey dilakukan untuk memvalidasi problem statement FaceFit Barber di lapangan")

    # Metrik survey
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responden Valid", "28")
    col2.metric("Pelanggan", "25")
    col3.metric("Kapster", "3")
    col4.metric("Pernah Kecewa", "64% (16/25)")
    st.markdown("")
    st.divider()

    # Baris 1: Frekuensi + Ketidakpuasan
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Frekuensi ke Barbershop (n=25 pelanggan)")
        fig, ax = plt.subplots(figsize=(6, 4))
        freq_colors = ['#90CAF9', '#42A5F5', '#1565C0']
        ax.pie(
            list(survey_freq.values()),
            labels=list(survey_freq.keys()),
            colors=freq_colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        ax.set_title('Frekuensi Kunjungan ke Barbershop')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### Pernah Kecewa dengan Hasil Potongan?")
        fig, ax = plt.subplots(figsize=(6, 4))
        disap_colors = ['#A5D6A7', '#FFF176', '#FFB74D', '#EF5350']
        bars = ax.bar(
            list(survey_disap.keys()),
            list(survey_disap.values()),
            color=disap_colors,
            edgecolor='white',
            linewidth=1
        )
        for bar, count in zip(bars, survey_disap.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(count), ha='center', fontsize=11, fontweight='bold')
        ax.set_title('Frekuensi Ketidakpuasan Pelanggan\n64% pernah kecewa (11+5 dari 25)')
        ax.set_ylabel('Jumlah Responden')
        ax.set_ylim(0, 14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Baris 2: Penyebab + Minat
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("### Penyebab Utama Ketidaksesuaian")
        fig, ax = plt.subplots(figsize=(6, 4))
        cause_colors = [warna_merah, warna_oren, '#FFC107', '#90A4AE']
        bars = ax.barh(
            list(survey_causes.keys()),
            list(survey_causes.values()),
            color=cause_colors,
            edgecolor='white'
        )
        for i, v in enumerate(survey_causes.values()):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Jumlah Responden')
        ax.set_title('Root Cause: Mengapa Pelanggan Kecewa?')
        ax.set_xlim(0, 12)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r2:
        st.markdown("### Minat Penggunaan Aplikasi (Skala 1–5)")
        fig, ax = plt.subplots(figsize=(6, 4))
        interest_colors = ['#EF9A9A', '#FFCC80', '#FFF59D', '#A5D6A7', '#66BB6A']
        bars = ax.bar(
            interest_scores,
            interest_counts,
            color=interest_colors,
            edgecolor='white',
            linewidth=1.5,
            width=0.6
        )
        for bar, count in zip(bars, interest_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(count), ha='center', fontsize=12, fontweight='bold')
        avg_score = sum(s * c for s, c in zip(interest_scores, interest_counts)) / sum(interest_counts)
        ax.axvline(avg_score, color='red', linestyle='--', linewidth=1.5, label=f'Rata-rata: {avg_score:.2f}/5')
        ax.set_title(f'Minat Pelanggan Pakai Aplikasi\nRata-rata: {avg_score:.2f}/5 — 68% beri skor ≥4')
        ax.set_xlabel('Skor Minat (1=Tidak tertarik, 5=Sangat tertarik)')
        ax.set_ylabel('Jumlah Pelanggan')
        ax.set_ylim(0, 18)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Perspektif Kapster
    st.markdown("### Perspektif Kapster (3 Responden)")
    col_k1, col_k2 = st.columns([2, 1])

    with col_k1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        kapster_names = ['Kapster 1\n(< 1 tahun)', 'Kapster 2\n(1-3 tahun)', 'Kapster 3\n(> 3 tahun)']
        x = range(3)
        w = 0.3
        ax.bar([i - w/2 for i in x], frek_kesulitan, w,
               label='Frekuensi kesulitan pelanggan', color=warna_merah, edgecolor='white')
        ax.bar([i + w/2 for i in x], minat_app_kapster, w,
               label='Minat pakai aplikasi', color=warna_hijau, edgecolor='white')
        for i, (fk, ma) in enumerate(zip(frek_kesulitan, minat_app_kapster)):
            ax.text(i - w/2, fk + 0.1, str(fk), ha='center', fontsize=10, fontweight='bold')
            ax.text(i + w/2, ma + 0.1, str(ma), ha='center', fontsize=10, fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels(kapster_names)
        ax.set_ylim(0, 6.5)
        ax.set_ylabel('Skor (1–5)')
        ax.set_title('Perspektif Kapster: Frekuensi Kesulitan vs Minat Aplikasi')
        ax.legend(fontsize=9)
        ax.axhline(3, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_k2:
        st.markdown("**Validasi dari Kapster:**")
        st.success("✅ Semua kapster mengakui pelanggan sering kesulitan mendeskripsikan keinginan")
        st.success("✅ Minat menggunakan aplikasi: rata-rata **4.67/5**")
        st.info("💡 Pain point ini yang menjadi landasan utama FaceFit Barber")

# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — PIPELINE & BALANCING
# ═══════════════════════════════════════════════════════════════════════════════
elif halaman == "⚙️ Pipeline & Balancing":

    st.subheader("Pipeline Data: Raw → Final (5,820 Gambar)")
    st.caption("Perjalanan data melalui cleaning, labeling MediaPipe, augmentasi, dan balancing")

    # Metrik pipeline
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Raw (filter Asian)", "39,098")
    col2.metric("Setelah Cleaning", "10,860")
    col3.metric("Setelah Labeling", "5,487")
    col4.metric("Setelah Augmentasi", "9,003")
    col5.metric("Dataset Final", "5,820 ✅")
    st.divider()

    # Imbalance ratio per tahap
    stage_list  = list(stages.keys())
    ir_list     = []
    total_list  = []
    for s in stage_list:
        vals = list(stages[s].values())
        ir_list.append(max(vals) / min(vals))
        total_list.append(sum(vals))

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Jumlah Gambar per Kelas di Setiap Tahap")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x  = np.arange(len(stage_list))
        w  = 0.15
        for i, cls in enumerate(CLASSES):
            vals = [stages[s][cls] for s in stage_list]
            ax.bar(x + i * w - 2 * w, vals, w,
                   label=cls, color=CLASS_COLORS[cls], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_list, fontsize=9)
        ax.set_title('Distribusi Kelas per Tahap Pipeline')
        ax.set_ylabel('Jumlah Gambar')
        ax.legend(title='Kelas', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### Imbalance Ratio per Tahap (BQ-4)")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ir_colors = [warna_merah if v > 1.5 else warna_hijau for v in ir_list]
        ax.bar(stage_list, ir_list, color=ir_colors, edgecolor='white', linewidth=1, width=0.5)
        ax.axhline(1.5, color='red', linestyle='--', linewidth=1.5, label='Target ≤ 1.5x (BQ-4)')
        for i, v in enumerate(ir_list):
            ax.text(i, v + 0.5, f'{v:.2f}x', ha='center', fontsize=10, fontweight='bold')
        ax.set_title('Imbalance Ratio per Tahap\n25.8x → 1.45x ✅')
        ax.set_ylabel('Imbalance Ratio (max/min)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Strategi Augmentasi
    st.markdown("### Strategi Augmentasi per Kelas")
    aug_data = pd.DataFrame([
        {'Kelas': 'oval',    'Gambar Asli': 4603, 'Gambar Baru': 0,   'Total': 4603, 'Strategi': 'Skip (di-undersample nanti)',                          'Variasi': '0x'},
        {'Kelas': 'round',   'Gambar Asli': 710,  'Gambar Baru': 390, 'Total': 1100, 'Strategi': 'Flip horizontal',                                      'Variasi': '1x'},
        {'Kelas': 'square',  'Gambar Asli': 671,  'Gambar Baru': 429, 'Total': 1100, 'Strategi': 'Flip horizontal',                                      'Variasi': '1x'},
        {'Kelas': 'heart',   'Gambar Asli': 413,  'Gambar Baru': 687, 'Total': 1100, 'Strategi': 'Flip H + brightness adjust',                           'Variasi': '2x'},
        {'Kelas': 'diamond', 'Gambar Asli': 178,  'Gambar Baru': 922, 'Total': 1100, 'Strategi': 'Flip H, rotate, brightness, contrast', 'Variasi': '6x'},
    ])

    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        fig, ax = plt.subplots(figsize=(9, 4))
        x  = np.arange(len(CLASSES))
        w  = 0.35
        ax.bar(x - w/2, aug_data['Gambar Asli'], w, color=warna_biru,  alpha=0.85, label='Gambar asli')
        ax.bar(x + w/2, aug_data['Gambar Baru'], w, color=warna_oren,  alpha=0.85, label='Gambar augmentasi baru')
        for i, row in aug_data.iterrows():
            if row['Gambar Baru'] > 0:
                ax.text(i + w/2, row['Gambar Baru'] + 15, row['Variasi'],
                        ha='center', fontsize=8.5, color='darkred')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES)
        ax.set_title('Gambar Asli vs Gambar Augmentasi Baru per Kelas')
        ax.set_ylabel('Jumlah Gambar')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_a2:
        st.dataframe(
            aug_data[['Kelas', 'Gambar Asli', 'Gambar Baru', 'Variasi', 'Strategi']],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Insight Diamond & Indo
    st.markdown("### Insight Penting dari Hasil Labeling")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("**Diamond: Hanya Ada di Dataset ZIP**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sources_d = ['UTKFace', 'FairFace', 'Indo\nPublic', 'Men Face\nShape', 'Dataset ZIP']
        diamond_c = [0, 0, 0, 0, 178]
        bar_col_d = ['#BDBDBD', '#BDBDBD', '#BDBDBD', '#BDBDBD', '#9C27B0']
        bars = ax.bar(sources_d, diamond_c, color=bar_col_d, edgecolor='white')
        ax.set_title('Gambar Diamond per Sumber', color='#9C27B0')
        ax.set_ylabel('Jumlah Gambar Diamond')
        ax.set_ylim(0, 230)
        ax.text(4, 185, '178', ha='center', fontsize=14, fontweight='bold', color='#9C27B0')
        for i in range(4):
            ax.text(i, 8, '0', ha='center', fontsize=12, color='gray')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_i2:
        st.markdown("**Indo Public Figure: 100% Oval**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        indo_count = [436, 0, 0, 0, 0]
        indo_col   = [CLASS_COLORS[c] for c in CLASSES]
        bars = ax.bar(CLASSES, indo_count, color=indo_col, edgecolor='white')
        ax.set_title('Distribusi Label Indo Public Figure', color='#4CAF50')
        ax.set_ylabel('Jumlah Gambar')
        ax.set_ylim(0, 520)
        ax.text(0, 445, '436', ha='center', fontsize=14, fontweight='bold', color='#4CAF50')
        for i in range(1, 5):
            ax.text(i, 8, '0', ha='center', fontsize=12, color='gray')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# HALAMAN 4 — DATASET FINAL
# ═══════════════════════════════════════════════════════════════════════════════
elif halaman == "✅ Dataset Final":

    st.subheader("Dataset Final — 5,820 Gambar Siap Training")
    st.caption("Output pipeline W2 yang diserahkan ke tim AI Engineer untuk training model CNN")

    # Metrik final
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Gambar", f"{total_final:,}")
    col2.metric("Jumlah Kelas", "5")
    col3.metric("Imbalance Ratio", f"{ir_final:.2f}x ✅")
    col4.metric("Status BQ-4", "TERCAPAI (≤ 1.5x)")
    st.divider()

    # Distribusi final
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Distribusi Kelas Dataset Final")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = [CLASS_COLORS[c] for c in CLASSES]
        bars = ax.bar(CLASSES, list(final_counts.values()), color=colors, edgecolor='white', linewidth=1)
        ax.set_title(f'Distribusi Dataset Final\n(total: {total_final:,} gambar — imbalance ratio: {ir_final:.2f}x)')
        ax.set_ylabel('Jumlah Gambar')
        ax.set_ylim(0, 1750)
        for bar, (cls, cnt) in zip(bars, final_counts.items()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{cnt:,}', ha='center', fontsize=11, fontweight='bold')
        ax.axhline(1.5 * min(final_counts.values()), color='red', linestyle='--',
                   linewidth=0.8, alpha=0.4, label=f'Batas 1.5x imbalance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("### Proporsi Kelas")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        colors = [CLASS_COLORS[c] for c in CLASSES]
        ax.pie(
            list(final_counts.values()),
            labels=[f'{cls}\n({cnt:,})' for cls, cnt in final_counts.items()],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        ax.set_title('Proporsi Kelas Dataset Final')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Kontribusi sumber per kelas
    st.markdown("### Kontribusi Sumber per Kelas (BQ-5)")
    source_composition = {
        'oval'   : {'UTKFace': 500, 'FairFace': 700, 'Indo Public': 300, 'Men Face Shape': 0,   'Dataset ZIP': 0},
        'round'  : {'UTKFace': 4,   'FairFace': 33,  'Indo Public': 0,   'Men Face Shape': 882, 'Dataset ZIP': 178},
        'square' : {'UTKFace': 6,   'FairFace': 20,  'Indo Public': 0,   'Men Face Shape': 898, 'Dataset ZIP': 173},
        'heart'  : {'UTKFace': 68,  'FairFace': 219, 'Indo Public': 0,   'Men Face Shape': 0,   'Dataset ZIP': 807},
        'diamond': {'UTKFace': 0,   'FairFace': 0,   'Indo Public': 0,   'Men Face Shape': 0,   'Dataset ZIP': 1032},
    }
    df_src = pd.DataFrame(source_composition).T
    df_src_pct = df_src.div(df_src.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    src_colors = [warna_biru, warna_hijau, warna_oren, '#9C27B0', warna_merah]
    df_src_pct.plot(kind='bar', stacked=True, ax=ax,
                    color=src_colors, edgecolor='white', linewidth=0.5, width=0.6)
    ax.axhline(70, color='red', linestyle='--', linewidth=1.2, label='Batas dominasi (70%) — BQ-5')
    ax.set_title('Kontribusi Sumber Dataset per Kelas Bentuk Wajah (%)', fontsize=13)
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Proporsi (%)')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Sumber', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.warning("⚠️ **Diamond 100% dari Dataset ZIP** — perlu tambahan sumber data lain di iterasi berikutnya untuk mengurangi risiko bias.")

    st.divider()

    # Tabel ringkasan + kesimpulan
    st.markdown("### Kesimpulan & Status Pertanyaan Bisnis")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        **BQ-1: Gap Komunikasi**  
        64% pelanggan (16/25) pernah tidak puas. Pain point utama: susah jelaskan keinginan (9 orang) dan gaya tidak cocok bentuk wajah (8 orang).
        """)
        st.success("✅ Teridentifikasi")
    with k2:
        st.markdown("""
        **BQ-3 & BQ-4: Kualitas Dataset**  
        Imbalance ratio final **1.45x** — di bawah target 1.5x. Dataset 5,820 gambar siap untuk training CNN oleh tim AI Engineer.
        """)
        st.success("✅ Imbalance 1.45x ≤ 1.5x")
    with k3:
        st.markdown("""
        **BQ-5: Dominasi Sumber**  
        Diamond 100% dari satu sumber (Dataset ZIP). Diperlukan penambahan data diamond dari sumber lain di iterasi selanjutnya.
        """)
        st.warning("⚠️ Perlu data diamond tambahan")

    st.divider()
    st.markdown("### Output yang Diserahkan ke Tim AI")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.info("""
        📁 **dataset_before_balancing/**  
        9,003 gambar — setelah augmentasi, sebelum undersample  
        Format: JPG, ukuran 224×224 px
        """)
    with col_o2:
        st.success("""
        📁 **dataset_after_balancing/**  
        5,820 gambar — dataset final siap training  
        Imbalance ratio: 1.45x ✅
        """)
