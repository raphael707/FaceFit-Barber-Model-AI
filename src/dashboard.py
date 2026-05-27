import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FaceFit Barber — Data Dashboard",
    page_icon="🧔",
    layout="wide"
)

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
warna_abu   = '#B0BEC5'

datasets_info = pd.DataFrame([
    {'Dataset': 'UTKFace',           'Total Raw': 66966, 'Masuk Pipeline': 9974,  'Filter': 'Asian only (race=2)',     'Label': 'MediaPipe'},
    {'Dataset': 'FairFace',          'Total Raw': 97698, 'Masuk Pipeline': 26047, 'Filter': 'East/Southeast Asian',   'Label': 'MediaPipe'},
    {'Dataset': 'Men Face Shape',    'Total Raw': 1309,  'Masuk Pipeline': 1004,  'Filter': 'Train split saja',       'Label': 'Sudah ada'},
    {'Dataset': 'Dataset ZIP',       'Total Raw': 1433,  'Masuk Pipeline': 1433,  'Filter': 'Hapus rect & triangle',  'Label': 'Sudah ada (5 kelas)'},
    {'Dataset': 'Indo Public Figure','Total Raw': 640,   'Masuk Pipeline': 640,   'Filter': '-',                      'Label': 'MediaPipe'},
])

labeling_data = {
    'UTKFace'         : {'oval': 1160, 'round': 4,   'square': 6,   'heart': 68,  'diamond': 0},
    'FairFace'        : {'oval': 3541, 'round': 33,  'square': 20,  'heart': 219, 'diamond': 0},
    'Indo Public'     : {'oval': 436,  'round': 0,   'square': 0,   'heart': 0,   'diamond': 0},
    'Men Face Shape'  : {'oval': 337,  'round': 338, 'square': 312, 'heart': 0,   'diamond': 0},
    'Dataset ZIP'     : {'oval': 356,  'round': 338, 'square': 338, 'heart': 167, 'diamond': 178},
}

stages = {
    'Sebelum\nAugmentasi': {'oval': 4603, 'round': 710,  'square': 671,  'heart': 413,  'diamond': 178},
    'Setelah\nAugmentasi': {'oval': 4603, 'round': 1100, 'square': 1100, 'heart': 1100, 'diamond': 1100},
    'Setelah\nValidasi'  : {'oval': 4601, 'round': 1097, 'square': 1097, 'heart': 1094, 'diamond': 1032},
    'Setelah\nBalancing' : {'oval': 1500, 'round': 1097, 'square': 1097, 'heart': 1094, 'diamond': 1032},
}

final_counts = {'oval': 1500, 'round': 1097, 'square': 1097, 'heart': 1094, 'diamond': 1032}
total_final  = sum(final_counts.values())
ir_final     = max(final_counts.values()) / min(final_counts.values())

# ── DATA SURVEY (dari file asli, 25 pelanggan valid + 3 kapster) ──────────────
survey_freq   = {'Sebulan sekali': 10, 'Jarang': 7, '2 bulan sekali': 5,
                 '3 bulan sekali': 2, '2-3 bulan sekali': 1}
survey_disap  = {'Tidak pernah': 3, 'Jarang': 6, 'Pernah beberapa kali': 11, 'Sering': 5}
survey_causes = {
    'Susah jelaskan\nke kapster'       : 9,
    'Gaya tidak cocok\nbentuk wajah'   : 8,
    'Referensi ≠\nhasil asli'          : 4,
    'Lainnya'                          : 4,
}
survey_prioritas = {
    'Kualitas hasil\npotongan': 22,
    'Harga\nterjangkau'       : 2,
    'Kecepatan\nlayanan'      : 1,
}
interest_scores  = [1, 2, 3, 4, 5]
interest_counts  = [5, 1, 2, 2, 15]   # n=25 pelanggan valid
frek_kesulitan   = [4, 3, 4]           # kapster (skor 1-5)
minat_app_kapster= [4, 5, 5]           # kapster (skor 1-5)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧔 FaceFit Barber")
    st.caption("Data Science Dashboard — CC26-PSU304")
    st.divider()
    halaman = st.radio(
        "Navigasi",
        ["Overview Dataset", "Hasil Survey", "Pipeline & Balancing", "Dataset Final"]
    )
    st.divider()
    st.markdown("""
    **Tim:** CC26-PSU304  
    **Path:** Data Scientist  
    **Dikerjakan oleh:**  
    Diwan Ramadhani Dwi Putra - CDCC119D6Y1045
    Rofiqho Izmi - CDCC295D6Y0838
    """)

st.title("FaceFit Barber — Dataset Dashboard")
st.caption("Analisis data pipeline W1–W3: dari raw dataset ke dataset siap training model CNN")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
if halaman == "Overview Dataset":

    st.subheader("Overview 5 Dataset yang Digunakan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dataset", "5 dataset")
    col2.metric("Total Raw (semua sumber)", f"{datasets_info['Total Raw'].sum():,}")
    col3.metric("Total Masuk Pipeline", f"{datasets_info['Masuk Pipeline'].sum():,}")
    col4.metric("Kelas Target", "5 kelas")
    st.markdown("")

    st.markdown("### Ringkasan Dataset")
    st.dataframe(datasets_info, use_container_width=True, hide_index=True)
    st.markdown("""
    Pipeline FaceFit menggunakan **5 dataset** dengan karakteristik berbeda. UTKFace dan FairFace adalah dataset wajah umum skala besar yang di-filter ke subpopulasi Asia untuk menjaga relevansi dengan target pengguna aplikasi. Men Face Shape dan Dataset ZIP sudah memiliki label bentuk wajah sehingga langsung dipakai tanpa proses labeling ulang. Indo Public Figure ditambahkan khusus untuk memperkuat representasi wajah lokal Indonesia.
    """)
    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Volume Raw vs Masuk Pipeline")
        fig, ax = plt.subplots(figsize=(9, 4))
        x  = np.arange(len(datasets_info))
        w  = 0.35
        b1 = ax.bar(x - w/2, datasets_info['Total Raw'],      color=warna_biru,  alpha=0.85, label='Total Raw')
        b2 = ax.bar(x + w/2, datasets_info['Masuk Pipeline'], color=warna_hijau, alpha=0.85, label='Masuk Pipeline')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_info['Dataset'], rotation=10, ha='right', fontsize=9)
        ax.set_ylabel('Jumlah Gambar')
        ax.set_title('Perbandingan Volume: Raw vs Masuk Pipeline')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
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
        st.markdown("""
        Dari total **168.046 gambar raw**, hanya sekitar **22,7%** (38.098 gambar) yang lolos masuk pipeline setelah proses filter ras. UTKFace dan FairFace memiliki volume raw terbesar namun tingkat filter paling tinggi karena mayoritas gambarnya bukan subpopulasi Asia. Sebaliknya, Dataset ZIP dan Indo Public Figure masuk 100% karena sudah relevan sejak awal.
        """)

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
        st.markdown("""
        FairFace mendominasi pipeline dengan **68,4%** kontribusi, diikuti UTKFace **26,2%**. Tiga dataset sisanya relatif kecil namun krusial — terutama Dataset ZIP sebagai **satu-satunya sumber kelas heart dan diamond**.
        """)

    st.divider()
    st.markdown("### Hasil Labeling MediaPipe per Sumber")
    st.caption("Diamond = 0 di UTKFace, FairFace, dan Indo Public Figure — hanya ada di Dataset ZIP")

    col_la, col_lb = st.columns([3, 2])

    with col_la:
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
        st.markdown("""
        Hasil labeling MediaPipe menunjukkan **ketimpangan ekstrem** antar kelas. UTKFace dan FairFace hampir seluruhnya menghasilkan label oval karena foto *in-the-wild* umumnya diambil dari sudut frontal yang membuat rasio tinggi/lebar wajah cenderung masuk threshold oval. Kelas diamond sama sekali tidak muncul di ketiga dataset besar tersebut — hanya Dataset ZIP yang menyediakannya, menjadikannya bottleneck utama pipeline.
        """)

    with col_lb:
        st.markdown("**Insight Kunci:**")
        st.info("**Diamond = 0** di UTKFace, FairFace, dan Indo Public. Wajah diamond langka di dataset *in-the-wild* Asia — threshold geometris MediaPipe sangat jarang terpenuhi.")
        st.warning("**Indo Public 100% Oval** — foto publik figur Indonesia sering diambil dari sudut 3/4 sehingga rasio height/width masuk threshold oval.")
        st.success("**Men Face Shape + Dataset ZIP** menjadi sumber utama untuk kelas round, square, heart, dan diamond.")

# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "Hasil Survey":

    st.subheader("Hasil Survey — 28 Responden (25 Pelanggan + 3 Kapster)")
    st.caption("Survey dilakukan untuk memvalidasi problem statement FaceFit Barber di lapangan")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responden Valid", "28")
    col2.metric("Pelanggan", "25")
    col3.metric("Kapster", "3")
    col4.metric("Pernah Kecewa", "64% (16/25)")
    st.markdown("")
    st.divider()

    # ── Row 1: Frekuensi + Kecewa ─────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Frekuensi ke Barbershop (n=25 pelanggan)")
        fig, ax = plt.subplots(figsize=(6, 4))
        freq_colors = ['#90CAF9', '#42A5F5', '#1976D2', '#0D47A1', '#B3E5FC']
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
        st.markdown("""
        Sebagian besar pelanggan datang ke barbershop **sebulan sekali atau lebih jarang** — menunjukkan bahwa setiap kunjungan adalah momen penting yang sayang jika berakhir dengan ketidakpuasan. Pola ini memperkuat urgensi solusi yang bisa membantu pelanggan mendapatkan hasil yang tepat sejak kali pertama.
        """)

    with col_r:
        st.markdown("### Pernah Kecewa dengan Hasil Potongan?")
        fig, ax = plt.subplots(figsize=(6, 4))
        disap_vals   = list(survey_disap.values())
        max_disap    = max(disap_vals)
        # Warna: highlight bar tertinggi, sisanya netral
        disap_colors = [warna_merah if v == max_disap else warna_abu for v in disap_vals]
        bars = ax.bar(
            list(survey_disap.keys()),
            disap_vals,
            color=disap_colors,
            edgecolor='white',
            linewidth=1
        )
        for bar, count in zip(bars, disap_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(count), ha='center', fontsize=11, fontweight='bold')
        ax.set_title('Frekuensi Ketidakpuasan Pelanggan\n64% pernah kecewa (11+5 dari 25)')
        ax.set_ylabel('Jumlah Responden')
        ax.set_ylim(0, 14)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        **64% pelanggan** (16 dari 25) pernah merasa hasil potongan tidak sesuai harapan — baik sesekali maupun sering. Hanya 3 orang yang belum pernah mengalami ketidakpuasan sama sekali. Angka ini mengkonfirmasi bahwa masalah mismatch di barbershop bukan pengecualian, melainkan pengalaman umum.
        """)

    st.divider()

    # ── Row 2: Penyebab + Minat ───────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("### Penyebab Utama Ketidaksesuaian")
        fig, ax = plt.subplots(figsize=(6, 4))
        cause_vals   = list(survey_causes.values())
        max_cause    = max(cause_vals)
        # Highlight bar terbesar (pain point #1), sisanya abu
        cause_colors = [warna_merah if v == max_cause else warna_abu for v in cause_vals]
        bars = ax.barh(
            list(survey_causes.keys()),
            cause_vals,
            color=cause_colors,
            edgecolor='white'
        )
        for i, v in enumerate(cause_vals):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=11, fontweight='bold')
        ax.set_xlabel('Jumlah Responden')
        ax.set_title('Root Cause: Mengapa Pelanggan Kecewa?')
        ax.set_xlim(0, 12)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        Dua penyebab teratas — **susah menjelaskan keinginan ke kapster** (9 orang) dan **gaya tidak cocok bentuk wajah** (8 orang) — secara langsung menjadi justifikasi utama FaceFit Barber. Keduanya bisa diselesaikan dengan deteksi bentuk wajah otomatis: pelanggan tidak perlu mendeskripsikan wajahnya, sistem yang mengenalinya.
        """)

    with col_r2:
        st.markdown("### Minat Penggunaan Aplikasi (Skala 1–5)")
        fig, ax = plt.subplots(figsize=(6, 4))
        max_interest = max(interest_counts)
        # Highlight skor 5 (tertinggi), sisanya abu
        int_colors = [warna_hijau if c == max_interest else warna_abu for c in interest_counts]
        bars = ax.bar(
            interest_scores,
            interest_counts,
            color=int_colors,
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
        st.markdown("""
        Distribusi skor minat sangat condong ke kanan — **15 dari 25 pelanggan memberi skor 5** (sangat tertarik). Rata-rata 3.84/5 dengan median 5 menunjukkan bahwa resistensi terhadap solusi digital ini sangat rendah di kalangan target pengguna.
        """)

    st.divider()

    # ── Row 3: Descriptive Stats + Prioritas ──────────────────────────────────
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        st.markdown("### Statistik Deskriptif — Minat Aplikasi (Pelanggan)")
        minat_series = pd.Series(
            [s for s, c in zip(interest_scores, interest_counts) for _ in range(c)],
            name='Skor Minat'
        )
        desc = minat_series.describe()
        stat_df = pd.DataFrame({
            'Statistik': ['Jumlah (n)', 'Rata-rata', 'Std Dev', 'Min', 'Q1 (25%)', 'Median', 'Q3 (75%)', 'Maks'],
            'Nilai'    : [
                int(desc['count']),
                f"{desc['mean']:.2f}",
                f"{desc['std']:.2f}",
                int(desc['min']),
                int(desc['25%']),
                int(desc['50%']),
                int(desc['75%']),
                int(desc['max']),
            ]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
        st.caption("Median = 5 dan Q1 = 3 menunjukkan distribusi condong ke kanan — mayoritas pelanggan sangat tertarik.")

    with col_r3:
        st.markdown("### Prioritas Utama Memilih Barbershop")
        fig, ax = plt.subplots(figsize=(6, 4))
        prio_vals   = list(survey_prioritas.values())
        max_prio    = max(prio_vals)
        prio_colors = [warna_biru if v == max_prio else warna_abu for v in prio_vals]
        bars = ax.bar(
            list(survey_prioritas.keys()),
            prio_vals,
            color=prio_colors,
            edgecolor='white',
            linewidth=1
        )
        for bar, v in zip(bars, prio_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    str(v), ha='center', fontsize=12, fontweight='bold')
        ax.set_title('Apa yang Paling Penting saat Memilih Barbershop?\n88% prioritaskan kualitas hasil potongan')
        ax.set_ylabel('Jumlah Responden')
        ax.set_ylim(0, 26)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        **88% pelanggan** menyebut kualitas hasil potongan sebagai faktor terpenting — jauh melampaui harga maupun lokasi. Ini memperkuat bahwa pelanggan barbershop adalah segmen yang *quality-conscious* dan bersedia menggunakan alat bantu selama hasilnya lebih baik.
        """)

    st.divider()

    # ── Row 4: Kapster ────────────────────────────────────────────────────────
    st.markdown("### Perspektif Kapster (3 Responden)")
    col_k1, col_k2 = st.columns([2, 1])

    with col_k1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        kapster_names = ['Kapster 1\n(< 1 tahun)', 'Kapster 2\n(1-3 tahun)', 'Kapster 3\n(1-3 tahun)']
        x = range(3)
        w = 0.3
        ax.bar([i - w/2 for i in x], frek_kesulitan,    w,
               label='Frekuensi kesulitan pelanggan', color=warna_merah, edgecolor='white')
        ax.bar([i + w/2 for i in x], minat_app_kapster, w,
               label='Minat pakai aplikasi',          color=warna_hijau, edgecolor='white')
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
        avg_minat_k = sum(minat_app_kapster) / len(minat_app_kapster)
        st.markdown("**Validasi dari Kapster:**")
        st.success("Semua kapster mengakui pelanggan sering kesulitan mendeskripsikan keinginan")
        st.success(f"Minat menggunakan aplikasi: rata-rata **{avg_minat_k:.2f}/5**")
        st.info("Pain point ini yang menjadi landasan utama FaceFit Barber")

# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "Pipeline & Balancing":

    st.subheader("Pipeline Data: Raw → Final (5,820 Gambar)")
    st.caption("Perjalanan data melalui cleaning, labeling MediaPipe, augmentasi, dan balancing")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Raw (filter Asian)", "39,098")
    col2.metric("Setelah Cleaning", "10,860")
    col3.metric("Setelah Labeling", "5,487")
    col4.metric("Setelah Augmentasi", "9,003")
    col5.metric("Dataset Final", "5,820")
    st.divider()

    stage_list = list(stages.keys())
    ir_list    = []
    total_list = []
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
        st.markdown("""
        Sebelum augmentasi, kelas oval mendominasi secara masif (~4.600 gambar) sementara diamond hanya 178. Augmentasi berhasil menaikkan semua kelas minoritas ke level 1.100, dan setelah validasi serta balancing, oval di-undersample ke 1.500 agar distribusi akhir jauh lebih seimbang. Pola "oval dominan → seimbang" ini adalah hasil utama dari seluruh pipeline.
        """)

    with col_r:
        st.markdown("### Imbalance Ratio per Tahap (BQ-4)")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ir_colors = [warna_merah if v > 1.5 else warna_hijau for v in ir_list]
        ax.bar(stage_list, ir_list, color=ir_colors, edgecolor='white', linewidth=1, width=0.5)
        ax.axhline(1.5, color='red', linestyle='--', linewidth=1.5, label='Target ≤ 1.5x (BQ-4)')
        for i, v in enumerate(ir_list):
            ax.text(i, v + 0.5, f'{v:.2f}x', ha='center', fontsize=10, fontweight='bold')
        ax.set_title('Imbalance Ratio per Tahap\n25.8x → 1.45x')
        ax.set_ylabel('Imbalance Ratio (max/min)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        Imbalance ratio awal sebesar **25.86x** sangat berbahaya untuk training model — artinya ada kelas yang 25 kali lebih banyak dari kelas lain, sehingga model berpotensi hanya "hafal" kelas dominan. Melalui augmentasi dan undersampling, ratio berhasil ditekan ke **1.45x**, di bawah target BQ-4 (≤1.5x). Ini artinya dataset sudah cukup seimbang untuk training yang fair.
        """)

    st.divider()

    st.markdown("### Strategi Augmentasi per Kelas")
    aug_data = pd.DataFrame([
        {'Kelas': 'oval',    'Gambar Asli': 4603, 'Gambar Baru': 0,   'Total': 4603, 'Strategi': 'Skip (di-undersample nanti)',           'Variasi': '0x'},
        {'Kelas': 'round',   'Gambar Asli': 710,  'Gambar Baru': 390, 'Total': 1100, 'Strategi': 'Flip horizontal',                       'Variasi': '1x'},
        {'Kelas': 'square',  'Gambar Asli': 671,  'Gambar Baru': 429, 'Total': 1100, 'Strategi': 'Flip horizontal',                       'Variasi': '1x'},
        {'Kelas': 'heart',   'Gambar Asli': 413,  'Gambar Baru': 687, 'Total': 1100, 'Strategi': 'Flip H + brightness adjust',            'Variasi': '2x'},
        {'Kelas': 'diamond', 'Gambar Asli': 178,  'Gambar Baru': 922, 'Total': 1100, 'Strategi': 'Flip H, rotate, brightness, contrast',  'Variasi': '6x'},
    ])

    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        fig, ax = plt.subplots(figsize=(9, 4))
        x  = np.arange(len(CLASSES))
        w  = 0.35
        ax.bar(x - w/2, aug_data['Gambar Asli'], w, color=warna_biru, alpha=0.85, label='Gambar asli')
        ax.bar(x + w/2, aug_data['Gambar Baru'], w, color=warna_oren, alpha=0.85, label='Gambar augmentasi baru')
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
        st.markdown("""
        Strategi augmentasi disesuaikan dengan tingkat kekurangan tiap kelas. Oval tidak diaugmentasi sama sekali karena sudah sangat banyak. Diamond mendapat augmentasi paling agresif — **6 variasi transformasi** — karena hanya tersedia 178 gambar asli. Semakin langka kelasnya, semakin beragam transformasi yang diterapkan untuk memaksimalkan variasi visual.
        """)

    with col_a2:
        st.dataframe(
            aug_data[['Kelas', 'Gambar Asli', 'Gambar Baru', 'Variasi', 'Strategi']],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.markdown("### Insight Penting dari Hasil Labeling")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("**Diamond: Hanya Ada di Dataset ZIP**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sources_d = ['UTKFace', 'FairFace', 'Indo\nPublic', 'Men Face\nShape', 'Dataset ZIP']
        diamond_c = [0, 0, 0, 0, 178]
        # Highlight satu-satunya sumber diamond
        bar_col_d = [warna_abu if v == 0 else '#9C27B0' for v in diamond_c]
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
        st.markdown("""
        Dari 5 dataset, hanya **Dataset ZIP** yang mengandung gambar diamond. Ini terjadi karena threshold geometris MediaPipe untuk kelas diamond (lebar tulang pipi lebih besar dari dahi sekaligus rahang) sangat jarang terpenuhi pada foto *in-the-wild* populasi Asia. Ketergantungan pada satu sumber ini menjadi risiko bias yang perlu diatasi di iterasi selanjutnya.
        """)
        st.markdown("**Indo Public Figure: 100% Oval**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        indo_count = [436, 0, 0, 0, 0]
        # Highlight oval saja
        indo_col = [CLASS_COLORS['oval'] if v > 0 else warna_abu for v in indo_count]
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
        st.markdown("""
        Seluruh 436 gambar Indo Public Figure dilabeli sebagai oval oleh MediaPipe. Dugaan utamanya adalah foto publik figur Indonesia kebanyakan diambil dari sudut 3/4 atau dengan sedikit tilt kepala, sehingga rasio dimensi wajah selalu masuk ke threshold oval. Dataset ini tetap berguna untuk memperkuat representasi wajah Asia Tenggara di kelas oval.
        """)

# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "Dataset Final":

    st.subheader("Dataset Final — 5,820 Gambar Siap Training")
    st.caption("Output pipeline W2 yang diserahkan ke tim AI Engineer untuk training model CNN")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Gambar", f"{total_final:,}")
    col2.metric("Jumlah Kelas", "5")
    col3.metric("Imbalance Ratio", f"{ir_final:.2f}x")
    col4.metric("Status BQ-4", "TERCAPAI (≤ 1.5x)")
    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Distribusi Kelas Dataset Final")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        # Highlight kelas terbesar (oval) dan terkecil (diamond)
        max_count = max(final_counts.values())
        min_count = min(final_counts.values())
        bar_colors = []
        for cls in CLASSES:
            if final_counts[cls] == max_count:
                bar_colors.append(warna_hijau)
            elif final_counts[cls] == min_count:
                bar_colors.append(warna_merah)
            else:
                bar_colors.append(warna_abu)
        bars = ax.bar(CLASSES, list(final_counts.values()), color=bar_colors, edgecolor='white', linewidth=1)
        ax.set_title(f'Distribusi Dataset Final\n(total: {total_final:,} gambar — imbalance ratio: {ir_final:.2f}x)')
        ax.set_ylabel('Jumlah Gambar')
        ax.set_ylim(0, 1750)
        for bar, (cls, cnt) in zip(bars, final_counts.items()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{cnt:,}', ha='center', fontsize=11, fontweight='bold')
        ax.axhline(1.5 * min(final_counts.values()), color='red', linestyle='--',
                   linewidth=0.8, alpha=0.4, label='Batas 1.5x imbalance')
        # Legend manual
        legend_patches = [
            mpatches.Patch(color=warna_hijau, label='Terbanyak (oval)'),
            mpatches.Patch(color=warna_merah, label='Tersedikit (diamond)'),
            mpatches.Patch(color=warna_abu,   label='Kelas lainnya'),
        ]
        ax.legend(handles=legend_patches, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        Dataset final terdiri dari **5.820 gambar** yang terdistribusi relatif merata di 5 kelas. Oval sengaja dibatasi di 1.500 (bukan lebih rendah lagi) untuk menjaga volume total tetap memadai bagi training. Diamond sebagai kelas terkecil memiliki 1.032 gambar — selisihnya dengan oval hanya 468 gambar, jauh lebih baik dibanding kondisi awal yang gap-nya lebih dari 4.000.
        """)

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

    st.markdown("### Kontribusi Sumber per Kelas (BQ-5)")
    source_composition = {
        'oval'   : {'UTKFace': 500, 'FairFace': 700, 'Indo Public': 300, 'Men Face Shape': 0,   'Dataset ZIP': 0},
        'round'  : {'UTKFace': 4,   'FairFace': 33,  'Indo Public': 0,   'Men Face Shape': 882, 'Dataset ZIP': 178},
        'square' : {'UTKFace': 6,   'FairFace': 20,  'Indo Public': 0,   'Men Face Shape': 898, 'Dataset ZIP': 173},
        'heart'  : {'UTKFace': 68,  'FairFace': 219, 'Indo Public': 0,   'Men Face Shape': 0,   'Dataset ZIP': 807},
        'diamond': {'UTKFace': 0,   'FairFace': 0,   'Indo Public': 0,   'Men Face Shape': 0,   'Dataset ZIP': 1032},
    }
    df_src     = pd.DataFrame(source_composition).T
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
    st.markdown("""
    Oval merupakan satu-satunya kelas yang memiliki kontribusi dari banyak sumber (UTKFace, FairFace, Indo Public), sehingga risikonya paling rendah. Round dan square bergantung dominan pada Men Face Shape, sedangkan heart dan diamond hampir sepenuhnya bergantung pada Dataset ZIP. Kelas diamond yang **100% dari satu sumber** adalah yang paling rentan terhadap bias dan perlu diperkuat di iterasi berikutnya.
    """)
    st.warning("**Diamond 100% dari Dataset ZIP** — perlu tambahan sumber data lain di iterasi berikutnya untuk mengurangi risiko bias.")

    st.divider()

    st.markdown("### Kesimpulan & Status Pertanyaan Bisnis")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("""
        **BQ-1: Gap Komunikasi**  
        64% pelanggan (16/25) pernah tidak puas. Pain point utama: susah jelaskan keinginan (9 orang) dan gaya tidak cocok bentuk wajah (8 orang).
        """)
        st.success("Teridentifikasi")
    with k2:
        st.markdown("""
        **BQ-3 & BQ-4: Kualitas Dataset**  
        Imbalance ratio final **1.45x** — di bawah target 1.5x. Dataset 5,820 gambar siap untuk training CNN oleh tim AI Engineer.
        """)
        st.success("Imbalance 1.45x ≤ 1.5x")
    with k3:
        st.markdown("""
        **BQ-5: Dominasi Sumber**  
        Diamond 100% dari satu sumber (Dataset ZIP). Diperlukan penambahan data diamond dari sumber lain di iterasi selanjutnya.
        """)
        st.warning("Perlu data diamond tambahan")

    st.divider()
    st.markdown("### Output yang Diserahkan ke Tim AI")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.info("""
        **dataset_before_balancing/**  
        9,003 gambar — setelah augmentasi, sebelum undersample  
        Format: JPG, ukuran 224×224 px
        """)
    with col_o2:
        st.success("""
        **dataset_after_balancing/**  
        5,820 gambar — dataset final siap training  
        Imbalance ratio: 1.45x
        """)
