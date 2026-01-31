#!/usr/bin/env python3
"""
plotting.py

Модуль визуализации для пайплайна FITS -> Report.
Создает единый график: Flux+Continuum и Residual с подсветкой полос.

API:
    plot_all(w_um, flux, continuum, residual, detections, out_png)

CLI:
    python plotting.py file.fits --out outputs/plot.png
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Импорт локальных модулей
try:
    import io_fits
    import continuum
    import features
except ImportError:
    # Если скрипт запускается из другой директории, пробуем добавить текущую
    sys.path.append(os.path.dirname(__file__))
    try:
        import io_fits
        import continuum
        import features
    except ImportError:
        print("Error: Could not import pipeline modules (io_fits, continuum, features).")
        sys.exit(1)

def plot_all(w_um, flux, continuum_val, residual, detections, out_png):
    """
    Создает и сохраняет финальный график.

    Args:
        w_um (array): Длины волн (мкм).
        flux (array): Поток.
        continuum_val (array): Континуум.
        residual (array): Нормализованный поток (flux/continuum).
        detections (dict): Результат detect_molecules.
        out_png (str): Путь для сохранения файла.
    """
    
    # --- PREMIUM STYLE CONFIGURATION ---
    # Dark mode with vibrant accents
    plt.style.use('dark_background')
    
    # Custom color palette (Cyberpunk/Neon)
    colors = {
        'flux': '#00FFFF',      # Cyan
        'cont': '#FF00FF',      # Magenta
        'resid': '#00FF00',     # Lime Green
        'grid': '#333333',
        'text': '#FFFFFF',
        'H2O': '#3399FF',       # Soft Blue
        'O2': '#FF3333',        # Neon Red
        'O3': '#FF9933',        # Neon Orange
        'bg_panes': '#121212'
    }
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.08)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # --- PANEL 1: SPECTRAL FLUX ---
    # Background for panes
    for ax in [ax1, ax2]:
        ax.set_facecolor(colors['bg_panes'])
        ax.grid(True, color=colors['grid'], linestyle='--', linewidth=0.5, alpha=0.6)
    
    # Plot Data
    ax1.step(w_um, flux, where='mid', color=colors['flux'], lw=1.2, alpha=0.9, label='Observed Flux')
    ax1.plot(w_um, continuum_val, color=colors['cont'], lw=2.5, alpha=0.9, linestyle='-', label='Model Continuum')
    
    # Decorate Panel 1
    ax1.set_ylabel('Flux Density [Jy or similar]', fontsize=11, fontweight='bold', color='white')
    ax1.tick_params(axis='both', colors='white', labelsize=10)
    
    # Legend with box
    leg = ax1.legend(loc='upper right', frameon=True, facecolor='#202020', edgecolor='gray', framealpha=0.9)
    for text in leg.get_texts(): text.set_color("white")
    
    # Title with Metadata
    ax1.set_title(f"SPECTRAL ANALYSIS: {os.path.basename(out_png)[:-9]}", 
                  fontsize=14, fontweight='bold', color='white', pad=15)

    # --- PANEL 2: RESIDUAL & FEATURES ---
    ax2.plot(w_um, residual, color=colors['resid'], lw=1.0, alpha=0.8, label='Residual')
    ax2.axhline(1.0, color='gray', linestyle='--', lw=1.5, alpha=0.5)
    
    # Dynamic Y-limits for residual
    res_clean = residual[np.isfinite(residual) & (residual > 0)]
    if len(res_clean) > 0:
        ymin, ymax = np.percentile(res_clean, [1, 99])
        yrange = ymax - ymin
        ax2.set_ylim(max(0, ymin - 0.3*yrange), ymax + 0.5*yrange)
    
    # Highlight Features
    trans = ax2.get_xaxis_transform()
    
    for mol, res in detections.items():
        c_mol = colors.get(mol, 'yellow')
        is_det = res['detected']
        
        # Track if we printed "No coverage" to avoid clutter?
        # Maybe just print it at the band center?
        
        for b in res['bands']:
            b_start, b_end = b['band_range']
            b_mid = (b_start + b_end) / 2
            
            if not b['covered']:
                # Label "No coverage" only if within plot x-limits?
                ax2.text(b_mid, 1.05, "No Coverage", ha='center', va='bottom', 
                         color='gray', fontsize=7, rotation=90, clip_on=True)
                continue
            
            # Fill Band
            alpha_val = 0.2 if is_det else 0.05
            ax2.axvspan(b_start, b_end, color=c_mol, alpha=alpha_val)
            
            # Label Peak Features
            if b['snr'] > 2.0: # Only label significant features
                 # Annotation text
                 note = f"{mol}\n-{b['depth']:.0%}"
                 # Position: at the bottom of the dip?
                 ax2.annotate(note, 
                              xy=(b_mid, 1.0 - b['depth']), 
                              xytext=(0, -15), textcoords="offset points",
                              ha='center', va='top', color=c_mol, fontsize=8, fontweight='bold',
                              arrowprops=dict(arrowstyle="-", color=c_mol, alpha=0.5))

    ax2.set_xlabel('Wavelength [μm]', fontsize=11, fontweight='bold', color='white')
    ax2.set_ylabel('Normalized Flux', fontsize=11, fontweight='bold', color='white')
    ax2.tick_params(axis='x', colors='white', labelsize=10)
    ax2.tick_params(axis='y', colors='white', labelsize=8)
    
    # Hide top xticks of ax2, bottom of ax1 handled by sharex but let's be sure
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    plt.tight_layout()
    
    # Создаем директорию выхода, если нужно
    out_dir = os.path.dirname(out_png)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Plot saved to {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Create unified spectral analysis plot.")
    parser.add_argument("file", help="Input FITS file")
    parser.add_argument("--out", required=True, help="Output PNG file path (e.g. outputs/plot.png)")
    args = parser.parse_args()
    
    # 1. Чтение
    try:
        data = io_fits.read_spectrum(args.file)
        w = data['wavelength_um']
        f = data['flux']
        e = data['err']
        meta = data['meta']
    except Exception as ex:
        print(f"Error reading {args.file}: {ex}")
        sys.exit(1)
        
    # 2. Континуум
    try:
        c_res = continuum.fit_blackbody(w, f, err=e)
        cont = c_res['continuum']
        residual = c_res['residual']
    except Exception as ex:
        print(f"Error fitting continuum: {ex}")
        sys.exit(1)
        
    # 3. Features (detections)
    try:
        detections = features.detect_molecules(w, residual, err=e)
    except Exception as ex:
        print(f"Error detecting features: {ex}")
        sys.exit(1)
        
    # 4. Plot
    try:
        plot_all(w, f, cont, residual, detections, args.out)
    except Exception as ex:
        print(f"Error plotting: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()
