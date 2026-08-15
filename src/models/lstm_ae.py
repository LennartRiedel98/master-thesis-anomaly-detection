"""LSTM-Autoencoder Detektor.

Sequenz-Autoencoder mit LSTM-Encoder/Decoder. Eingang sind gleitende
Fenster der Laenge ``window_size`` ueber die Feature-Matrix. Der Score
einer Stunde wird aus dem Rekonstruktions-Fehler des Fensters gebildet,
das auf dieser Stunde *endet*. Die ersten ``window_size - 1`` Stunden
sowie Fenster mit NaN bekommen NaN als Score.

Score-Varianten (``score_mode``, Ablation zur Kriteriums-Mismatch-
Diagnose, s. methodology 7.12; Training/Repraesentation bleiben
identisch, nur die Fehler->Score-Abbildung aendert sich):
    window_mse    (Default, bisheriges Verhalten): MSE ueber das ganze
                  Fenster und alle Features - mittelt punktuelle
                  Abweichungen ueber W x F Werte und verwaessert sie.
    channel_mse   MSE nur auf dem Ziel-Kanal (erste Feature-Spalte =
                  kw_mean/residual/trend je Variante), gemittelt ueber
                  das Fenster - entfernt die Verduennung durch die
                  Kalender-/Wetter-Features.
    last_step_mse MSE nur auf dem letzten Zeitschritt (alle Features) -
                  punktschaerfste Attribution zur Score-Stunde.
    mahalanobis   EncDec-AD-Scoring (Malhotra u. a. 2016): Fehlervektor
                  des letzten Zeitschritts, bewertet als Mahalanobis-
                  Distanz zur Fehlerverteilung N(mu, Sigma) der
                  Trainingsdaten. mu/Sigma werden am Ende von fit()
                  geschaetzt (bei Alt-Checkpoints: fit_error_stats()).

Hyperparameter (vgl. Methodik 7.5):
    window_size:    Fensterlaenge in Stunden (z. B. 24 = Tagesfenster)
    n_layers:       Anzahl gestapelter LSTM-Schichten
    hidden_size:    Einheiten pro Schicht
    learning_rate:  Adam-LR
    batch_size:     Mini-Batch
    epochs:         Trainings-Epochen
    score_mode:     s. oben; Default "window_mse" (Hauptergebnisse
                    unveraendert)
    device:         "auto" | "cpu" | "cuda" | "mps" - bei "auto" wird die
                    beste verfuegbare GPU gewaehlt: CUDA (z. B. RTX 3080 Ti
                    mobile) vor MPS (Apple-Metal, z. B. M4 Pro), sonst CPU.
    random_state:   Seed fuer Torch/Numpy (CUDA ist nicht voll determi-
                    nistisch - siehe Reproduzierbarkeits-Hinweis in der
                    Methodik).

``torch`` wird lazy importiert, damit ``models.registry`` und die
klassischen Detektoren auch auf Maschinen ohne torch-Installation
nutzbar bleiben (z. B. der Mac fuer die Stages 1-6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import AnomalyDetector

# Cache fuer den lazy torch-Import + die nn.Module-Definition. Die
# Modulklasse kann erst definiert werden, wenn torch importiert ist;
# wir bauen sie beim ersten Bedarf und merken sie hier.
_TORCH_CACHE: dict[str, Any] = {}


def _torch():
    """Lazy-Import von torch + Definition des Seq2Seq-Autoencoders."""
    if "torch" not in _TORCH_CACHE:
        import torch
        from torch import nn

        class _Seq2SeqAE(nn.Module):
            """Encoder verdichtet das Fenster zu einem Latent-Vektor;
            der Decoder rekonstruiert daraus die gesamte Sequenz."""

            def __init__(self, input_size: int, hidden_size: int,
                         n_layers: int, window_size: int) -> None:
                """Encoder-, Decoder-LSTM und die Ausgabeschicht anlegen.

                Der Decoder arbeitet auf der Breite ``hidden_size`` und wird erst von
                der linearen Schicht zurueck auf die Merkmalsdimension gebracht.
                """
                super().__init__()
                self.window_size = window_size
                self.encoder = nn.LSTM(input_size, hidden_size, n_layers,
                                       batch_first=True)
                self.decoder = nn.LSTM(hidden_size, hidden_size, n_layers,
                                       batch_first=True)
                self.out = nn.Linear(hidden_size, input_size)

            def forward(self, x):                 # x: (B, W, F)
                """Fenster kodieren, Latent-Vektor wiederholen, Sequenz rekonstruieren.

                Der verdichtende Schritt ist ``h[-1]``: Vom Encoder bleibt nur der
                versteckte Zustand der letzten Schicht am Fensterende uebrig - das ist
                der Flaschenhals. Er wird ueber die Fensterlaenge wiederholt und dient
                dem Decoder als Eingang. Genau dieser Engpass zwingt das Netz, nur die
                wiederkehrenden Regelmaessigkeiten zu speichern; was er nicht abbildet,
                taucht als Rekonstruktionsfehler auf.

                Formen: Eingang (B, W, F), Ausgang ebenfalls (B, W, F).
                """
                _, (h, _c) = self.encoder(x)
                latent = h[-1]                    # (B, hidden) letzter Layer
                rep = latent.unsqueeze(1).repeat(1, self.window_size, 1)
                dec, _ = self.decoder(rep)        # (B, W, hidden)
                return self.out(dec)              # (B, W, F)

        _TORCH_CACHE["torch"] = torch
        _TORCH_CACHE["nn"] = nn
        _TORCH_CACHE["module"] = _Seq2SeqAE
    return _TORCH_CACHE["torch"], _TORCH_CACHE["nn"], _TORCH_CACHE["module"]


class LSTMAutoencoderDetector(AnomalyDetector):
    """LSTM-Autoencoder: Anomalie = was das Netz nicht rekonstruieren kann.

    Der einzige Deep-Learning-Vertreter der Arbeit und in keinem der neun
    Jobs Champion. Die Diagnostik zeigt, dass das Netz die Tages- und
    Wochenmuster durchaus lernt - der Bruch sitzt in der Abbildung vom
    Rekonstruktionsfehler auf den Score, nicht in der Repraesentation.
    """

    name = "lstm_ae"

    def __init__(self,
                 window_size: int = 24,
                 n_layers: int = 2,
                 hidden_size: int = 32,
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 50,
                 device: str = "auto",
                 random_state: int = 42,
                 score_mode: str = "window_mse",
                 **hparams: Any) -> None:
        """Hyperparameter setzen und die spaeter gefitteten Felder anlegen.

        ``device='auto'`` waehlt CUDA, wenn verfuegbar, sonst CPU - MPS wird
        bewusst nicht automatisch genommen (auf dem Entwicklungs-Mac war es fuer
        dieses Netz unbrauchbar). ``score_mode`` legt fest, wie aus dem
        Rekonstruktionsfehler ein Score wird; die Score-Ablation vergleicht die
        vier Varianten am selben trainierten Modell.
        """
        super().__init__(window_size=window_size,
                         n_layers=n_layers,
                         hidden_size=hidden_size,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         epochs=epochs,
                         device=device,
                         random_state=random_state,
                         score_mode=score_mode,
                         **hparams)
        self.feature_cols_: list[str] | None = None
        self.input_size_: int | None = None
        self._model = None       # nn.Module
        self._device: str | None = None
        # Fehlerverteilungs-Statistik fuer score_mode="mahalanobis"
        # (EncDec-AD): Mittelwert und Praezisionsmatrix (Inverse der
        # Kovarianz) der Last-Step-Fehlervektoren auf den Trainingsdaten.
        self.err_mu_: np.ndarray | None = None
        self.err_prec_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """Loest device="auto" zur besten verfuegbaren Backend-Wahl auf.

        Praeferenz: CUDA (NVIDIA, z. B. RTX 3080 Ti am Trainings-Laptop)
        vor MPS (Apple-Metal-GPU, z. B. M4 Pro mit nativem arm64-Python)
        vor CPU. Ein explizit gesetztes device ("cpu"/"cuda"/"mps") wird
        unveraendert durchgereicht.
        """
        dev = self.hparams["device"]
        if dev != "auto":
            return dev
        torch, _, _ = _torch()
        if torch.cuda.is_available():
            return "cuda"
        # getattr-Guard: aeltere torch-Builds kennen torch.backends.mps nicht.
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _make_windows(self, arr: np.ndarray):
        """Gleitende Fenster (m, W, F) + zugehoerige End-Indizes.

        Nur Fenster ohne NaN werden zurueckgegeben; das End-Index-Array
        verweist auf die Position der letzten Stunde jedes Fensters im
        Original-Array.
        """
        w = self.hparams["window_size"]
        n = len(arr)
        if n < w:
            return (np.empty((0, w, arr.shape[1]), dtype=np.float32),
                    np.empty((0,), dtype=int))
        sw = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)
        # sliding_window_view liefert (n-w+1, F, W) -> auf (n-w+1, W, F).
        sw = sw.transpose(0, 2, 1)
        ends = np.arange(w - 1, n)
        valid = ~np.isnan(sw).any(axis=(1, 2))
        return sw[valid].astype(np.float32), ends[valid]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame) -> "LSTMAutoencoderDetector":
        """Autoencoder auf den Fenstern der Trainingsdaten trainieren.

        Die Reihe wird in gleitende Fenster der Laenge ``window_size``
        zerschnitten; trainiert wird auf Rekonstruktion des jeweiligen Fensters.
        Fenster mit fehlenden Werten fallen heraus.

        Die Startwerte fuer torch und numpy werden fest gesetzt. Auf der CPU ist
        das Ergebnis damit exakt reproduzierbar, auf CUDA bleibt eine Restvarianz
        durch nicht-deterministische Kernel - deshalb ist die Seed-Streuung
        eigens untersucht und als Limitation gefuehrt.
        """
        torch, nn, Module = _torch()
        self.feature_cols_ = list(X.columns)
        self.input_size_ = X.shape[1]
        self._device = self._resolve_device()

        # Reproduzierbarkeit (CPU exakt; CUDA bis auf nicht-deterministische
        # Kernel - bewusst in Kauf genommen, siehe Methodik).
        seed = self.hparams["random_state"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        windows, _ends = self._make_windows(X.to_numpy(dtype=float))
        if len(windows) == 0:
            raise ValueError(
                "LSTMAutoencoderDetector.fit: keine vollstaendigen Fenster "
                f"(window_size={self.hparams['window_size']})."
            )

        device = self._device
        model = Module(self.input_size_, self.hparams["hidden_size"],
                       self.hparams["n_layers"],
                       self.hparams["window_size"]).to(device)
        opt = torch.optim.Adam(model.parameters(),
                               lr=self.hparams["learning_rate"])
        loss_fn = nn.MSELoss()

        # float32 statt float64: NN-Standard, halbiert den Speicher und ist
        # auf Apple-MPS nativ. float64 auf MPS erzwingt pro Operation einen
        # CPU-Fallback (Hin-/Her-Kopieren), was den Speicher ueber die vielen
        # Trainings-Iterationen massiv aufblaeht. Die Modellgewichte sind
        # ohnehin float32 - so passt auch der Eingabe-dtype dazu.
        data = torch.from_numpy(windows.astype(np.float32)).to(device)
        n_samples = data.shape[0]
        bs = self.hparams["batch_size"]

        model.train()
        for _epoch in range(self.hparams["epochs"]):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, bs):
                idx = perm[start:start + bs]
                batch = data[idx]
                opt.zero_grad()
                recon = model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()

        self._model = model
        self.fitted = True
        # Fehlerverteilung fuer das Mahalanobis-Scoring direkt mitschaetzen
        # (ein zusaetzlicher Inferenz-Pass ueber die Trainingsfenster -
        # vernachlaessigbar gegen die Trainingszeit).
        self.fit_error_stats(X)
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _raw_errors(self, X: pd.DataFrame):
        """Ein Inferenz-Pass: Fehler-Bausteine je Fenster.

        Rueckgabe: (ends, window_mse, channel_mse, last_step_mse,
        last_err_vec) - last_err_vec ist der rohe Fehlervektor des
        letzten Zeitschritts (m, F) fuer die Mahalanobis-Bewertung.
        Ziel-Kanal = erste Feature-Spalte (Konvention aus
        models.registry.model_features: per-WMZ-Spalten zuerst, also
        kw_mean/residual/trend je Variante).
        """
        self._check_fitted()
        torch, _, _ = _torch()
        device = self._device or self._resolve_device()

        windows, ends = self._make_windows(
            X[self.feature_cols_].to_numpy(dtype=float))
        m = len(windows)
        w_mse = np.empty(m); c_mse = np.empty(m); l_mse = np.empty(m)
        l_vec = np.empty((m, self.input_size_), dtype=np.float64)
        if m == 0:
            return ends, w_mse, c_mse, l_mse, l_vec

        self._model.eval()
        bs = self.hparams["batch_size"]
        with torch.no_grad():
            data = torch.from_numpy(windows).to(device)
            for s in range(0, m, bs):
                batch = data[s:s + bs]
                diff = self._model(batch) - batch          # (B, W, F)
                sq = diff ** 2
                sl = slice(s, s + len(batch))
                w_mse[sl] = sq.mean(dim=(1, 2)).cpu().numpy()
                c_mse[sl] = sq[:, :, 0].mean(dim=1).cpu().numpy()
                l_mse[sl] = sq[:, -1, :].mean(dim=1).cpu().numpy()
                l_vec[sl] = diff[:, -1, :].cpu().numpy()
        return ends, w_mse, c_mse, l_mse, l_vec

    def fit_error_stats(self, X: pd.DataFrame) -> "LSTMAutoencoderDetector":
        """Schaetzt mu/Sigma^-1 der Last-Step-Fehlervektoren (EncDec-AD).

        Wird von fit() automatisch mit den Trainingsdaten aufgerufen.
        Fuer Alt-Checkpoints (vor score_mode) einmalig manuell mit den
        *Trainings*-Zeilen nachziehen - nie mit Test-Daten (Leakage).
        """
        _ends, _w, _c, _l, vec = self._raw_errors(X)
        if len(vec) == 0:
            raise ValueError("fit_error_stats: keine vollstaendigen Fenster.")
        self.err_mu_ = vec.mean(axis=0)
        cov = np.cov(vec, rowvar=False).reshape(self.input_size_,
                                                self.input_size_)
        # Ridge-Regularisierung vor der (Pseudo-)Inversen: die Kalender-
        # Features (sin/cos) sind untereinander stark korreliert, die
        # Kovarianz sonst nahe singulaer.
        eps = 1e-6 * max(np.trace(cov) / self.input_size_, 1e-12)
        self.err_prec_ = np.linalg.pinv(cov + eps * np.eye(self.input_size_))
        return self

    def score_components(self, X: pd.DataFrame) -> pd.DataFrame:
        """Alle Score-Varianten in einem Pass (fuer die Ablation 7.12).

        Spalten: window_mse, channel_mse, last_step_mse, mahalanobis
        (Letztere NaN, wenn keine Fehler-Statistik vorliegt).
        """
        ends, w_mse, c_mse, l_mse, vec = self._raw_errors(X)
        out = pd.DataFrame(np.nan, index=X.index,
                           columns=["window_mse", "channel_mse",
                                    "last_step_mse", "mahalanobis"])
        if len(ends) == 0:
            return out
        out.iloc[ends, 0] = w_mse
        out.iloc[ends, 1] = c_mse
        out.iloc[ends, 2] = l_mse
        if self.err_mu_ is not None:
            d = vec - self.err_mu_
            out.iloc[ends, 3] = np.einsum("ij,jk,ik->i", d,
                                          self.err_prec_, d)
        return out

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Score je Stunde nach dem eingestellten ``score_mode`` zurueckgeben.

        Berechnet alle vier Varianten ueber ``score_components`` und gibt die
        gewaehlte zurueck. ``mahalanobis`` setzt voraus, dass die
        Fehlerverteilung bekannt ist; bei Checkpoints aus der Zeit vor dieser
        Erweiterung muss ``fit_error_stats`` einmal nachgeholt werden.
        """
        mode = self.hparams.get("score_mode", "window_mse")
        comp = self.score_components(X)
        if mode not in comp.columns:
            raise ValueError(f"Unbekannter score_mode: {mode!r} "
                             f"(erlaubt: {list(comp.columns)})")
        if mode == "mahalanobis" and self.err_mu_ is None:
            raise RuntimeError(
                "score_mode='mahalanobis' ohne Fehler-Statistik - bei "
                "Alt-Checkpoints zuerst fit_error_stats(X_train) aufrufen.")
        return comp[mode].rename("score")

    def save(self, path: Path) -> None:
        """Gewichte, Hyperparameter und Fehlerstatistik als torch-Checkpoint ablegen.

        Die Gewichte werden vor dem Schreiben auf die CPU geholt, damit sich der
        Checkpoint auch auf einer Maschine ohne GPU laden laesst.
        """
        self._check_fitted()
        torch, _, _ = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "hparams": self.hparams,
            "feature_cols": self.feature_cols_,
            "input_size": self.input_size_,
            "state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
            # Fehlerverteilung (Mahalanobis-Scoring); None bei Alt-Staenden.
            "err_mu": self.err_mu_,
            "err_prec": self.err_prec_,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "LSTMAutoencoderDetector":
        """Detektor aus einem Checkpoint rekonstruieren und in den Eval-Modus setzen.

        Das Netz wird aus den gespeicherten Hyperparametern neu aufgebaut und
        mit den Gewichten befuellt. ``err_mu``/``err_prec`` fehlen in aelteren
        Checkpoints und bleiben dann None - alle Score-Modi ausser
        ``mahalanobis`` funktionieren trotzdem.
        """
        torch, _, Module = _torch()
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**state["hparams"])
        obj.feature_cols_ = state["feature_cols"]
        obj.input_size_ = state["input_size"]
        # Alt-Checkpoints (vor score_mode) haben keine Fehler-Statistik;
        # .get() laesst sie None -> fit_error_stats() bei Bedarf.
        obj.err_mu_ = state.get("err_mu")
        obj.err_prec_ = state.get("err_prec")
        obj._device = obj._resolve_device()
        model = Module(obj.input_size_, obj.hparams["hidden_size"],
                       obj.hparams["n_layers"],
                       obj.hparams["window_size"]).to(obj._device)
        model.load_state_dict(state["state_dict"])
        model.eval()
        obj._model = model
        obj.fitted = True
        return obj
