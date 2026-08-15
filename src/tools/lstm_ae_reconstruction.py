"""LSTM-AE-Rekonstruktion sichtbar machen und mit der MSTL-Zerlegung vergleichen.

Der LSTM-Autoencoder lernt Muster *implizit*: Der Flaschenhals zwingt ihn,
nur die wiederkehrenden Regelmaessigkeiten (Tagesgang, Wochenrhythmus,
Temperaturkopplung) zu kodieren - sie liegen verteilt in den Gewichten,
nicht als benennbare Komponente. Dieses Skript macht das gelernte
Normalmodell einsehbar und stellt es dem *expliziten* Normalmodell der
MSTL (Trend + Saison_24h + Saison_168h) gegenueber:

    Abb. 1 (immer):
      (a) Zeitreihen-Overlay ueber einen waehlbaren Ausschnitt:
          Original-kW, AE-Rekonstruktion, MSTL-Fit (nur Variante raw -
          fuer residual/trend ist der Kanal selbst schon eine
          MSTL-Komponente, ein Fit-Overlay waere zirkulaer).
      (b) Mittleres Tagesprofil derselben Reihen ueber den Ausschnitt -
          zeigt auf einen Blick, ob der AE den Tagesgang gelernt hat.
    Abb. 2 (--latent):
      PCA der Latent-Vektoren (Encoder-Endzustand h[-1] je Fenster),
      eingefaerbt nach Stunde und Wochentag. Ordnen sich die Punkte
      danach, hat der AE die Saisonalitaet intern repraesentiert.

Diagnostischer Zweck (Kap. 4 Diagnostik / Diskussion 5.6): Rekonstruiert
der AE das Tagesmuster sauber und trennt trotzdem Anomalien schlecht,
liegt das Problem im Scoring/der Schwelle (Kriteriums-Mismatch); lernt er
schon das Muster nicht, ist der Befund fundamentaler.

Rekonstruktions-Zuordnung: Wie beim Score wird jeder Stunde das Fenster
zugeordnet, das auf ihr *endet*; geplottet wird der letzte Zeitschritt
dieses Fensters. Die ersten ``window_size - 1`` Stunden bleiben NaN.

Eingang: outputs/<ds>/models/<variant>/<wmz>/lstm_ae.pkl   (Stage 7/8)
         outputs/<ds>/parquet/stage6_normalized_<variant>.parquet
         outputs/<ds>/scalers/scaler_<variant>.parquet
         outputs/<ds>/parquet/stage3_stl.parquet            (nur raw)
Ausgabe: outputs/<ds>/figures/lstm_ae_reconstruction_<variant>_<wmz>.png
         outputs/<ds>/figures/lstm_ae_latent_<variant>_<wmz>.png  (--latent)
         outputs/<ds>/reports/lstm_ae_reconstruction_<variant>_<wmz>.csv

Laeuft auf CUDA (RTX), MPS (Mac) oder CPU - Checkpoints sind portabel
(state_dict wird beim Speichern auf CPU gelegt, s. models/lstm_ae.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Direktstart als Skript: sys.path[0] ist src/tools/, die Modell-Imports
# liegen aber unter src/. Gleiches Muster wie score_distributions.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.lstm_ae import LSTMAutoencoderDetector, _torch  # noqa: E402
from models.registry import WMZ_NAMES, model_features       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "demo_synthetic"

# Feste Farbzuordnung (nicht am Serienrang, an der Rolle): Original neutral
# grau, AE-Rekonstruktion blau, MSTL-Fit orange. Blau/Orange ist auch unter
# Farbfehlsichtigkeit trennbar; der MSTL-Fit traegt zusaetzlich einen
# eigenen Linienstil als Zweitkodierung.
C_ORIG, C_AE, C_MSTL = "#555555", "C0", "C1"


def parse_args() -> argparse.Namespace:
    """Kommandozeilenargumente einlesen."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"Unterordner unter outputs/. Default: {DEFAULT_DATASET}")
    p.add_argument("--variant", default="raw",
                   choices=["raw", "residual", "trend"],
                   help="Signalebene des Checkpoints. Default: raw")
    p.add_argument("--wmz", default="wmz_2", choices=WMZ_NAMES,
                   help="Zaehler. Default: wmz_2 (staerkste Tagesperiodik)")
    p.add_argument("--start", default="2023-09-04",
                   help="Beginn des Plot-Ausschnitts (liegt im Test-Split; "
                        "Default: 2023-09-04, ein Montag)")
    p.add_argument("--days", type=int, default=14,
                   help="Laenge des Ausschnitts in Tagen. Default: 14")
    p.add_argument("--latent", action="store_true",
                   help="Zusaetzlich PCA-Plot der Latent-Vektoren erzeugen")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Inferenz-Batch. Default: 256")
    return p.parse_args()


def load_inputs(out_root: Path, variant: str, wmz: str):
    """Checkpoint + Feature-Frame + Scaler laden; X wie in Stage 7 slicen."""
    ckpt = out_root / "models" / variant / wmz / "lstm_ae.pkl"
    if not ckpt.is_file():
        raise SystemExit(
            f"Kein LSTM-AE-Checkpoint: {ckpt}\n"
            f"Run (RTX): python src/stage7_train.py --variants {variant} "
            f"--wmz {wmz} --models lstm_ae")
    det = LSTMAutoencoderDetector.load(ckpt)

    df = pd.read_parquet(out_root / "parquet"
                         / f"stage6_normalized_{variant}.parquet")
    if "split" not in df.columns:
        df = df.join(pd.read_parquet(out_root / "parquet"
                                     / "split_assignment.parquet"))
    X = model_features(df, variant, wmz)
    # Spalten exakt in Trainings-Reihenfolge - der Checkpoint ist die
    # Wahrheit, falls sich model_features seit dem Training geaendert hat.
    X = X[det.feature_cols_]

    scaler = pd.read_parquet(out_root / "scalers" / f"scaler_{variant}.parquet")
    return det, df, X, scaler


def reconstruct(det: LSTMAutoencoderDetector, X: pd.DataFrame,
                batch_size: int):
    """Per-Stunde-Rekonstruktion + Latent-Vektoren.

    Rueckgabe: (recon_df, latent_arr, ends) - recon_df enthaelt je Stunde
    den letzten Zeitschritt des auf ihr endenden Fensters (alle Features,
    normalisierter Raum); latent_arr die zugehoerigen h[-1]-Vektoren.
    """
    torch, _, _ = _torch()
    device = det._device or "cpu"
    windows, ends = det._make_windows(X.to_numpy(dtype=float))
    if len(windows) == 0:
        raise SystemExit("Keine vollstaendigen Fenster im gewaehlten Frame.")

    model = det._model
    model.eval()
    recon_last = np.empty((len(windows), windows.shape[2]), dtype=np.float32)
    latent = np.empty((len(windows), model.encoder.hidden_size),
                      dtype=np.float32)
    with torch.no_grad():
        data = torch.from_numpy(windows).to(device)
        for s in range(0, len(windows), batch_size):
            batch = data[s:s + batch_size]
            _, (h, _c) = model.encoder(batch)
            lat = h[-1]
            rep = lat.unsqueeze(1).repeat(1, model.window_size, 1)
            dec, _ = model.decoder(rep)
            out = model.out(dec)                      # (B, W, F)
            recon_last[s:s + len(batch)] = out[:, -1, :].cpu().numpy()
            latent[s:s + len(batch)] = lat.cpu().numpy()

    recon_df = pd.DataFrame(np.nan, index=X.index, columns=X.columns,
                            dtype=float)
    recon_df.iloc[ends] = recon_last
    return recon_df, latent, ends


def denorm(series: pd.Series, scaler: pd.DataFrame, col: str) -> pd.Series:
    """z-Score-Normalisierung invertieren: x * std + mean (Stage 6)."""
    return series * scaler.loc[col, "std"] + scaler.loc[col, "mean"]


def main() -> None:
    """Rekonstruktion des LSTM-AE zeichnen und der MSTL-Zerlegung gegenueberstellen.

    Macht sichtbar, was das Netz gelernt hat. Der Autoencoder kodiert die
    wiederkehrenden Regelmaessigkeiten implizit - verteilt in den Gewichten,
    nicht als benennbare Komponente. Die Gegenueberstellung mit dem
    *expliziten* Normalmodell der MSTL (Trend + Tages- + Wochensaison) zeigt,
    dass beide dasselbe Muster erfassen. Mit ``--latent`` kommt eine
    PCA-Projektion des Latent-Raums dazu, in der sich die Punkte nach
    Tageszeit ordnen.

    Das ist der Beleg dafuer, dass die Schwaeche des LSTM-AE nicht in der
    Repraesentation liegt.
    """
    args = parse_args()
    out_root = ROOT / "outputs" / args.dataset
    fig_dir = out_root / "figures"
    rep_dir = out_root / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    det, df, X, scaler = load_inputs(out_root, args.variant, args.wmz)
    print(f"Checkpoint: {args.variant}/{args.wmz}/lstm_ae.pkl  "
          f"(window={det.hparams['window_size']}, "
          f"hidden={det.hparams['hidden_size']}, device={det._device})")

    recon_df, latent, ends = reconstruct(det, X, args.batch_size)

    # Ziel-Kanal = erste per-WMZ-Spalte (raw: kw_mean, residual: residual,
    # trend: trend) - so haengt der Plot nicht an variantenspezifischen
    # Namen. In physikalische Einheiten zurueckskalieren.
    channel = det.feature_cols_[0]
    orig = denorm(X[channel], scaler, channel)
    recon = denorm(recon_df[channel], scaler, channel)

    # MSTL-Fit nur fuer raw: explizites Normalmodell Trend + beide
    # Saisonkomponenten aus Stage 3 (physikalische Einheiten).
    mstl_fit = None
    if args.variant == "raw":
        stl = pd.read_parquet(out_root / "parquet" / "stage3_stl.parquet")
        mstl_fit = (stl[f"{args.wmz}_trend"]
                    + stl[f"{args.wmz}_seasonal_24h"]
                    + stl[f"{args.wmz}_seasonal_168h"])

    # ------------------------------------------------------ Kennzahlen
    err = (recon - orig)
    split = df["split"]
    print("\nRekonstruktionsfehler (Ziel-Kanal, physikalisch, RMSE):")
    for sp in ["train", "val", "test"]:
        e = err[split == sp].dropna()
        print(f"  {sp:5s}: {np.sqrt((e ** 2).mean()):8.4f}  (n={len(e)})")
    both = pd.concat([recon.rename("ae"), orig.rename("orig")], axis=1).dropna()
    r_orig = both["ae"].corr(both["orig"])
    print(f"  r(AE-Rekonstruktion, Original)  = {r_orig:.3f}")
    if mstl_fit is not None:
        both = pd.concat([recon.rename("ae"), mstl_fit.rename("mstl")],
                         axis=1).dropna()
        print(f"  r(AE-Rekonstruktion, MSTL-Fit) = "
              f"{both['ae'].corr(both['mstl']):.3f}")

    # ------------------------------------------------------ Abb. 1
    start = pd.Timestamp(args.start)
    end = start + pd.Timedelta(days=args.days)
    if start < X.index.min() or start >= X.index.max():
        raise SystemExit(f"--start {args.start} liegt ausserhalb der Daten "
                         f"({X.index.min()} .. {X.index.max()}).")
    sl = slice(start, end)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]})

    unit = "kW" if args.variant == "raw" else "kW (Komponente)"
    ax1.plot(orig.loc[sl], color=C_ORIG, lw=1.0, label="Original")
    ax1.plot(recon.loc[sl], color=C_AE, lw=1.6,
             label="LSTM-AE-Rekonstruktion (implizites Normalmodell)")
    if mstl_fit is not None:
        ax1.plot(mstl_fit.loc[sl], color=C_MSTL, lw=1.6, ls="--",
                 label="MSTL-Fit: Trend + Saison 24 h + 168 h "
                       "(explizites Normalmodell)")
    ax1.set_ylabel(f"Leistung [{unit}]")
    ax1.set_title(f"Gelerntes vs. zerlegtes Normalmodell - {args.wmz}, "
                  f"Variante {args.variant}\n"
                  f"Ausschnitt {start.date()} bis {end.date()} "
                  f"({split.loc[sl].mode().iat[0]}-Split)", pad=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Mittleres Tagesprofil ueber den Ausschnitt: gelernt vs. zerlegt.
    for series, color, ls, lab in [
            (orig, C_ORIG, "-", "Original"),
            (recon, C_AE, "-", "LSTM-AE"),
            (mstl_fit, C_MSTL, "--", "MSTL-Fit")]:
        if series is None:
            continue
        prof = series.loc[sl].groupby(series.loc[sl].index.hour).mean()
        ax2.plot(prof.index, prof.to_numpy(), color=color, ls=ls, lw=1.8,
                 label=lab)
    ax2.set_xlabel("Stunde des Tages")
    ax2.set_ylabel(f"Mittel [{unit}]")
    ax2.set_title("Mittleres Tagesprofil des Ausschnitts", pad=8)
    ax2.set_xticks(range(0, 24, 3))
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out_png = fig_dir / (f"lstm_ae_reconstruction_{args.variant}"
                         f"_{args.wmz}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Wrote {out_png}")

    # ------------------------------------------------------ Report-CSV
    rep = pd.DataFrame({"original": orig, "ae_reconstruction": recon,
                        "error": err, "split": split})
    if mstl_fit is not None:
        rep["mstl_fit"] = mstl_fit
    csv_path = rep_dir / (f"lstm_ae_reconstruction_{args.variant}"
                          f"_{args.wmz}.csv")
    rep.to_csv(csv_path)
    print(f"  Wrote {csv_path}")

    # ------------------------------------------------------ Abb. 2 (opt.)
    if args.latent:
        from sklearn.decomposition import PCA
        # Ausduennen haelt den Scatter lesbar (jedes 3. Fenster genuegt,
        # die Nachbarfenster unterscheiden sich nur um eine Stunde).
        sel = np.arange(0, len(latent), 3)
        pcs = PCA(n_components=2, random_state=0).fit_transform(latent[sel])
        ts = X.index[ends[sel]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True,
                                 sharey=True)
        for ax, vals, cmap, lab, ticks in [
                (axes[0], ts.hour, "twilight", "Stunde des Tages",
                 range(0, 24, 6)),
                (axes[1], ts.weekday, "viridis", "Wochentag (0 = Mo)",
                 range(7))]:
            sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=vals, cmap=cmap, s=6,
                            alpha=0.6, linewidths=0)
            fig.colorbar(sc, ax=ax, label=lab, ticks=list(ticks))
            ax.set_xlabel("PC 1")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("PC 2")
        fig.suptitle(f"Latent-Raum des LSTM-AE (PCA) - {args.wmz}, "
                     f"Variante {args.variant}: Ordnung nach Tageszeit/"
                     f"Wochentag = implizit gelernte Saisonalitaet")
        fig.tight_layout()
        out_png = fig_dir / f"lstm_ae_latent_{args.variant}_{args.wmz}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_png}")


if __name__ == "__main__":
    main()
