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

def plot_all(w_um, flux, continuum_val, residual, detections, out_png,
             err=None, T_K=None, target_name=None):
    """
    Создает и сохраняет финальный график.

    Args:
        w_um (array): Длины волн (мкм).
        flux (array): Поток.
        continuum_val (array): Континуум.
        residual (array): Нормализованный поток (flux/continuum).
        detections (dict): Результат detect_molecules.
        out_png (str): Путь для сохранения файла.
        err (array, optional): Погрешности потока для shaded region.
        T_K (float, optional): Температура континуума для info-box.
        target_name (str, optional): Имя цели для заголовка.
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
        'H2O': '#3399FF',       # Blue
        'O2': '#FF3333',        # Red
        'O3': '#FF9933',        # Orange
        'CH4': '#CC44FF',       # Purple
        'CO':  '#FFDD00',       # Yellow
        'CO2': '#FF6699',       # Pink
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

    # Shaded error region
    if err is not None:
        e = np.asarray(err, dtype=float)
        valid_e = np.isfinite(flux) & np.isfinite(e) & (e > 0)
        ax1.fill_between(w_um[valid_e],
                         (flux - e)[valid_e], (flux + e)[valid_e],
                         alpha=0.22, color=colors['flux'], linewidth=0, label='±1σ error')

    # Molecule band shading on top panel (light background)
    for mol, res in detections.items():
        c_mol = colors.get(mol, 'yellow')
        for b in res['bands']:
            if b['covered'] and b.get('snr', 0) > 1.0:
                b_start, b_end = b['band_range']
                ax1.axvspan(b_start, b_end, color=c_mol, alpha=0.07, linewidth=0)

    # Decorate Panel 1
    ax1.set_ylabel('Flux Density', fontsize=11, fontweight='bold', color='white')
    ax1.tick_params(axis='both', colors='white', labelsize=10)

    # Legend
    leg = ax1.legend(loc='upper left', frameon=True, facecolor='#202020', edgecolor='gray', framealpha=0.9)
    for text in leg.get_texts(): text.set_color("white")

    # Title
    _title = target_name or os.path.splitext(os.path.basename(out_png))[0]
    ax1.set_title(f"SPECTRAL ANALYSIS: {_title}",
                  fontsize=14, fontweight='bold', color='white', pad=15)

    # Info-box (top-right): temperature + detection summary
    info_lines = []
    if T_K is not None and T_K > 0:
        info_lines.append(f"T = {T_K:.0f} K")
    info_lines.append("─" * 18)
    for mol, res in detections.items():
        status = res.get('status', 'NOT DETECTED')
        snr = res.get('max_snr', 0.0)
        if status == 'NO SPECTRAL COVERAGE':
            info_lines.append(f"{mol:<4}  NO COVERAGE")
        else:
            info_lines.append(f"{mol:<4}  {status:<10}  SNR={snr:.1f}")
    if info_lines:
        ax1.text(0.985, 0.97, "\n".join(info_lines),
                 transform=ax1.transAxes, ha='right', va='top',
                 fontsize=8.5, color='white', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                           edgecolor='#555555', alpha=0.88))

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
    for mol, res in detections.items():
        c_mol = colors.get(mol, 'yellow')
        is_det = res['detected']

        for b in res['bands']:
            b_start, b_end = b['band_range']
            b_mid = (b_start + b_end) / 2

            if not b['covered']:
                # Small dashed vertical line instead of cluttered rotated text
                ax2.axvline(b_mid, color='gray', lw=0.6, ls=':', alpha=0.35)
                continue

            # Fill Band
            alpha_val = 0.22 if is_det else 0.06
            ax2.axvspan(b_start, b_end, color=c_mol, alpha=alpha_val)

            # Annotate significant features with mol name, depth and SNR
            if b['snr'] > 2.0:
                note = f"{mol}  d={b['depth']:.2f}  SNR={b['snr']:.1f}"
                y_tip = max(res_clean.min() if len(res_clean) else 0.5,
                            1.0 - b['depth'] * 0.8)
                ax2.annotate(note,
                             xy=(b_mid, y_tip),
                             xytext=(0, -18), textcoords="offset points",
                             ha='center', va='top', color=c_mol,
                             fontsize=7.5, fontweight='bold',
                             arrowprops=dict(arrowstyle="-", color=c_mol, alpha=0.4))

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
        detections = features.detect_molecules(w, residual)
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
