# %%
# ==============================================================================
# PIPELINE ISTITUZIONALE DI REPLICA DINAMICA: OLS STATICA vs HMM-KALMAN FILTER
# Execution: Strictly Causal, Walk-Forward Nested CV, Zero Look-Ahead Bias
# ==============================================================================

import warnings
from hmmlearn.hmm import GaussianHMM
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

# Configurazione ambiente e soppressione log selettiva per Optuna
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 11
warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
# ==============================================================================
# MODULO 1: INGESTIONE DATI & COSTRUZIONE PANIERI (SENZA TITOLI CORROTTI)
# ==============================================================================
percorso_file_dati = "Dataset3_PortfolioReplicaStrategyErrataCorrige.xlsx"
percorso_file_meta = "Dataset3_PortfolioReplicaStrategy.xlsx"

# Estrazione metadati e ticker da foglio statico
nomi_completi = (
    pd.read_excel(percorso_file_meta, header=None, skiprows=3, nrows=1)
    .iloc[0]
    .tolist()[1:]
)
ticker_bloomberg = (
    pd.read_excel(percorso_file_meta, header=None, skiprows=5, nrows=1)
    .iloc[0]
    .tolist()[1:]
)

# Caricamento serie storiche grezze
dati_grezzi = pd.read_excel(percorso_file_dati, header=None, skiprows=1)
dati_grezzi.columns = ["Data"] + ticker_bloomberg
dati_grezzi["Data"] = pd.to_datetime(dati_grezzi["Data"], format="%d/%m/%Y")
dati_prezzi = dati_grezzi.set_index("Data").sort_index()

# Rimozione definitiva di LLL1 Comdty per assenza di dati esogeni affidabili
if "LLL1 Comdty" in dati_prezzi.columns:
    dati_prezzi = dati_prezzi.drop(columns=["LLL1 Comdty"])

# Calcolo rendimenti discreti semplici: R_t = (P_t / P_{t-1}) - 1
rendimenti = dati_prezzi.pct_change().dropna()

# Costruzione del Benchmark Target (Indice Composito Multi-Asset)
componenti_indice_target = {
    "HFRXGL Index": 0.50,  # Hedge Fund Research Global Index - 50%
    "MXWO Index": 0.25,  # MSCI World Equity Index - 25%
    "LEGATRUU Index": 0.25,  # Bloomberg Global Aggregate Bond Index - 25%
}

rendimento_target = pd.Series(
    0.0, index=rendimenti.index, name="Indice_Target"
)
for componente, peso in componenti_indice_target.items():
    rendimento_target += rendimenti[componente] * peso

# Definizione dell'universo di replica: 10 contratti future altamente liquidi
contratti_future = [
    "RX1 Comdty",
    "TY1 Comdty",
    "GC1 Comdty",
    "CO1 Comdty",
    "ES1 Comdty",
    "VG1 Comdty",
    "NQ1 Comdty",
    "TP1 Comdty",
    "DU1 Comdty",
    "TU2 Comdty",
]

# Allineamento rigoroso tra target (Y) e predittori (C)
date_comuni = rendimento_target.index.intersection(rendimenti.index)
Y = rendimento_target.loc[date_comuni]
C = rendimenti.loc[date_comuni, contratti_future]

print("--- STATO INGESTIONE DATI ---")
print(
    f"Osservazioni totali: {len(Y)} settimane | Universo predittori: {C.shape[1]}"
    " future"
)
print(
    f"Periodo di analisi: dal {Y.index.min().date()} al {Y.index.max().date()}\n"
)

# %%
# ==============================================================================
# MODULO 2: PARTIZIONAMENTO CRONOLOGICO RIGOROSO (60% TRAIN / 20% VAL / 20% TEST)
# ==============================================================================
T_totale = len(Y)
idx_train = int(T_totale * 0.60)
idx_val = int(T_totale * 0.80)

# Segregazione temporale assoluta (Nessun mescolamento casuale)
Y_train, C_train = Y.iloc[:idx_train], C.iloc[:idx_train]
Y_val, C_val = Y.iloc[idx_train:idx_val], C.iloc[idx_train:idx_val]
Y_test, C_test = Y.iloc[idx_val:], C.iloc[idx_val:]

print("--- PARTIZIONAMENTO TEMPORALE DEI BLOCCHI ---")
print(
    f"Training Set   (00% - 60%): {len(Y_train)} obs | dal"
    f" {Y_train.index.min().date()} al {Y_train.index.max().date()}"
)
print(
    f"Validation Set (60% - 80%): {len(Y_val)} obs  | dal"
    f" {Y_val.index.min().date()} al {Y_val.index.max().date()} (OOS 1)"
)
print(
    f"Test Set       (80% -100%): {len(Y_test)} obs  | dal"
    f" {Y_test.index.min().date()} al {Y_test.index.max().date()} (OOS 2)\n"
)

# %%
# ==============================================================================
# MODULO 3: BASELINE OLS STATICA (ZERO LOOK-AHEAD BIAS)
# ==============================================================================
# Addestramento pesi OLS esclusivamente in-sample (Training Set)
# Omissione intercetta per forzare la replica pura tramite combinazione di asset
modello_ols = LinearRegression(fit_intercept=False).fit(
    C_train.values, Y_train.values
)
pesi_ols_statici = pd.Series(
    modello_ols.coef_, index=contratti_future, name="Peso_OLS"
)

# Proiezione fuori campione (Out-of-Sample) passiva
rendimenti_ols_totali = pd.Series(
    modello_ols.predict(C.values), index=C.index, name="Replica_OLS"
)
rendimenti_ols_train = rendimenti_ols_totali.iloc[:idx_train]
rendimenti_ols_val = rendimenti_ols_totali.iloc[idx_train:idx_val]
rendimenti_ols_test = rendimenti_ols_totali.iloc[idx_val:]

esposizione_lorda_ols = PesOls = np.abs(pesi_ols_statici).sum()
print("--- PESI STIMATI BASELINE OLS STATICA ---")
print(pesi_ols_statici.round(4))
print(f"Esposizione Lorda Statica (Gross Exposure): {esposizione_lorda_ols:.4f}\n")

# %%
# ==============================================================================
# MODULO 4: IDENTIFICAZIONE CAUSALE DEI REGIMI DI MERCATO (GAUSSIAN HMM)
# ==============================================================================
# Fit del Modello Hidden Markoviano a 3 stati ESCLUSIVAMENTE sul Training Set
X_train_hmm = Y_train.values.reshape(-1, 1)
modello_hmm = GaussianHMM(
    n_components=3, covariance_type="full", n_iter=1000, random_state=42
).fit(X_train_hmm)


def prevedi_regimi_causali(modello, serie_temporale):
    """Calcolo Forward-Only dei regimi (Probabilità Alpha) per evitare il look-ahead

    bias intrinseco all'algoritmo di smoothing globale di Viterbi.
    """
    X = serie_temporale.values.reshape(-1, 1)
    T = len(X)
    log_prob_osservazione = modello._compute_log_likelihood(X)
    log_prob_iniziale = np.log(modello.startprob_ + 1e-10)
    log_matrice_transizione = np.log(modello.transmat_ + 1e-10)

    regimi_causali = np.zeros(T, dtype=int)
    buffer_lavoro = log_prob_iniziale + log_prob_osservazione[0]
    regimi_causali[0] = np.argmax(buffer_lavoro)

    for t in range(1, T):
        max_log = np.max(buffer_lavoro)
        alpha_precedente = np.exp(buffer_lavoro - max_log)
        prob_transizione = np.dot(alpha_precedente, modello.transmat_)
        buffer_lavoro = (
            np.log(prob_transizione + 1e-10)
            + max_log
            + log_prob_osservazione[t]
        )
        regimi_causali[t] = np.argmax(buffer_lavoro)

    return regimi_causali


# Mappatura regimi ordinata per volatilità crescente (0=Calma, 1=Volatilità, 2=Crisi)
serie_regimi_grezza = pd.Series(prevedi_regimi_causali(modello_hmm, Y), index=Y.index)
volatilita_per_regime = Y_train.groupby(serie_regimi_grezza.iloc[:idx_train]).std().sort_values()
mappa_ordinamento = {
    vecchio_id: nuovo_id
    for nuovo_id, vecchio_id in enumerate(volatilita_per_regime.index)
}
serie_regimi_causali = serie_regimi_grezza.map(mappa_ordinamento).astype(int)

print("--- DISTRIBUZIONE DEI REGIMI CAUSALI (TRAINING SET) ---")
for reg_id, nome_reg in enumerate(
    ["Calma (0)", "Volatilità (1)", "Crisi (2)"]
):
    conteggio = (serie_regimi_causali.iloc[:idx_train] == reg_id).sum()
    print(
        f"Regime {nome_reg}: {conteggio} settimane"
        f" ({conteggio/len(Y_train)*100:.1f}%)"
    )
print("")

# %%
# ==============================================================================
# MODULO 5: FILTRO DI KALMAN ISTITUZIONALE & META-OTTIMIZZAZIONE OPTUNA
# ==============================================================================
class ReplicatoreKalmanIstituzionale:
    """Filtro di Kalman a commutazione di regime per la replica dinamica di portafoglio.

    I pesi a priori w_{t|t-1} operano come variabili di stato esecutibili senza look-ahead.
    L'aggiornamento bayesiano avviene strettamente sui segnali di mercato puri.
    """

    def __init__(self, dev_std_Q_regimi, dev_std_R_regimi):
        self.dev_std_Q_regimi = dev_std_Q_regimi
        self.dev_std_R_regimi = dev_std_R_regimi

    def addestra_e_filtra(self, Y_valori, C_valori, regimi, alfa_ridge=1.0):
        T, N = C_valori.shape
        self.pesi_exante = np.zeros((T, N))
        self.rendimenti_previsti_puri = np.zeros(T)

        # Inizializzazione rigorosa via Ridge sui primi 3 mesi (12 settimane) del Train Set
        stimatore_ridge = Ridge(alpha=alfa_ridge, fit_intercept=False).fit(
            C_valori[:12], Y_valori[:12]
        )
        x_precedente = stimatore_ridge.coef_

        # Covarianza di stato P_{0|0} proporzionale alla varianza residua in-sample
        varianza_residua = np.var(
            Y_valori[:12] - C_valori[:12] @ x_precedente
        )
        P_precedente = np.eye(N) * (
            varianza_residua / (np.var(C_valori[:12]) + 1e-6)
        )

        for t in range(T):
            regime_corrente = int(regimi[t])
            Q = np.eye(N) * (
                self.dev_std_Q_regimi.get(regime_corrente, 0.01) ** 2
            )
            R = self.dev_std_R_regimi.get(regime_corrente, 0.01) ** 2

            C_t = C_valori[t].reshape(1, -1)
            y_t = Y_valori[t]

            # 1. PASSO DI PREDIZIONE EX-ANTE (A Priori: x_{t|t-1}, P_{t|t-1})
            x_predetto = x_precedente
            P_predetto = P_precedente + Q
            y_hat_t = (C_t @ x_predetto).item()

            # 2. AGGIORNAMENTO BAYESIANO A POSTERIORI (Innovazione pura senza attriti)
            S = (C_t @ P_predetto @ C_t.T).item() + R
            K = (P_predetto @ C_t.T) / S  # Guadagno di Kalman: Shape (N, 1)

            innovazione = y_t - y_hat_t
            x_aggiornato = x_predetto + K.flatten() * innovazione

            # Stabilità numerica della covarianza via Forma di Joseph
            I_KC = np.eye(N) - K @ C_t
            P_aggiornato = I_KC @ P_predetto @ I_KC.T + (K @ K.T) * R

            # Registrazione variabili operative ex-ante
            self.rendimenti_previsti_puri[t] = y_hat_t
            self.pesi_exante[t] = x_predetto

            x_precedente, P_precedente = x_aggiornato, P_aggiornato

        return self


# --- META-OTTIMIZZAZIONE WALK-FORWARD NESTED CV SUL SOLO TRAINING SET ---
print(
    "Avvio ottimizzazione Optuna (Walk-Forward CV 4-Folds strettamente nel Train Set"
    " 0%-60%)..."
)


def funzione_obiettivo_optuna(trial):
    dev_std_Q = {
        i: trial.suggest_float(f"Q_{i}", 1e-5, 0.1, log=True) for i in range(3)
    }
    dev_std_R = {
        i: trial.suggest_float(f"R_{i}", 1e-5, 0.1, log=True) for i in range(3)
    }

    Y_train_vals = Y_train.values
    C_train_vals = C_train.values
    regimi_train_vals = serie_regimi_causali.iloc[:idx_train].values

    # Convalida incrociata per serie storiche a finestre espansive (Expanding Window)
    cv_temporale = TimeSeriesSplit(
        n_splits=4, test_size=int(len(Y_train) * 0.15)
    )
    errori_tracking_folds = []

    for idx_sub_train, idx_sub_val in cv_temporale.split(Y_train_vals):
        ind_fine = idx_sub_val[-1] + 1
        modello_cv = ReplicatoreKalmanIstituzionale(
            dev_std_Q_regimi=dev_std_Q, dev_std_R_regimi=dev_std_R
        )
        modello_cv.addestra_e_filtra(
            Y_train_vals[:ind_fine],
            C_train_vals[:ind_fine],
            regimi_train_vals[:ind_fine],
        )

        # Valutazione Tracking Error esclusivamente sulla finestra di convalida del fold
        predizioni_val = modello_cv.rendimenti_previsti_puri[idx_sub_val]
        valori_veri_val = Y_train_vals[idx_sub_val]
        te_fold = np.std(predizioni_val - valori_veri_val) * np.sqrt(52)
        errori_tracking_folds.append(te_fold)

    return np.mean(errori_tracking_folds)


studio_optuna = optuna.create_study(direction="minimize")
studio_optuna.optimize(funzione_obiettivo_optuna, n_trials=120)

miglior_Q = {
    0: studio_optuna.best_params["Q_0"],
    1: studio_optuna.best_params["Q_1"],
    2: studio_optuna.best_params["Q_2"],
}
miglior_R = {
    0: studio_optuna.best_params["R_0"],
    1: studio_optuna.best_params["R_1"],
    2: studio_optuna.best_params["R_2"],
}

print(
    f"Ottimizzazione terminata. Tracking Error medio in CV interna:"
    f" {studio_optuna.best_value*100:.2f}%"
)
print(f"Iperparametri Ottimi Q (Rumore di Processo): {miglior_Q}")
print(f"Iperparametri Ottimi R (Rumore di Osservazione): {miglior_R}\n")

# --- ESECUZIONE GLOBALE DEL MODELLO KALMAN CON IPERPARAMETRI OTTIMI ---
modello_kalman_ottimo = ReplicatoreKalmanIstituzionale(
    dev_std_Q_regimi=miglior_Q, dev_std_R_regimi=miglior_R
)
modello_kalman_ottimo.addestra_e_filtra(
    Y.values, C.values, serie_regimi_causali.values
)

df_pesi_kalman = pd.DataFrame(
    modello_kalman_ottimo.pesi_exante, index=Y.index, columns=C.columns
)
rendimenti_kalman_grezzi = pd.Series(
    modello_kalman_ottimo.rendimenti_previsti_puri, index=Y.index
)

# --- CALCOLO ESATTO DEL TURNOVER EX-POST CON DRIFT DEI PESI DI PORTAFOGLIO ---
rendimenti_kalman_netti = pd.Series(
    0.0, index=Y.index, name="Replica_Kalman_Netta"
)
serie_turnover = pd.Series(0.0, index=Y.index, name="Turnover")
costo_transazione_unitario = 0.0005  # 5 bps per esecuzione su future liquidi

for t in range(len(Y)):
    if t == 0:
        rendimenti_kalman_netti.iloc[t] = rendimenti_kalman_grezzi.iloc[t]
        serie_turnover.iloc[t] = np.abs(df_pesi_kalman.iloc[t]).sum()
    else:
        # Pesi pre-rebalancing mutati per l'evoluzione dei prezzi sottostanti (Drift)
        pesi_t_meno_1 = df_pesi_kalman.iloc[t - 1]
        rendimento_asset_t = C.iloc[t]
        pesi_con_drift = (pesi_t_meno_1 * (1 + rendimento_asset_t)) / (
            1 + rendimenti_kalman_grezzi.iloc[t]
        )

        # Turnover effettivo addebitabile alla clearing house al tempo t
        turnover_t = np.abs(df_pesi_kalman.iloc[t] - pesi_con_drift).sum()
        serie_turnover.iloc[t] = turnover_t

        # Detrazione dell'attrito di esecuzione dal rendimento di replica realizzato
        rendimenti_kalman_netti.iloc[t] = (
            rendimenti_kalman_grezzi.iloc[t]
            - turnover_t * costo_transazione_unitario
        )

# Segregazione definitiva delle performance (Out-of-Sample 1 e 2 rigorosamente isolati)
rendimenti_kf_train = rendimenti_kalman_netti.iloc[:idx_train]
rendimenti_kf_val = rendimenti_kalman_netti.iloc[idx_train:idx_val]
rendimenti_kf_test = rendimenti_kalman_netti.iloc[idx_val:]

pesi_kf_train = df_pesi_kalman.iloc[:idx_train]
pesi_kf_val = df_pesi_kalman.iloc[idx_train:idx_val]
pesi_kf_test = df_pesi_kalman.iloc[idx_val:]

# %%
# ==============================================================================
# MODULO 6: MOTORE DI VALUTAZIONE PERFORMANCE & TABELLE MULTI-INDICE
# ==============================================================================
def calcola_metriche_istituzionali(
    r_target, r_replica, pesi_asset=None, e_target=False, freq=52.0
):
    """Calcola le 8 metriche chiave per un portafoglio istituzionale su una finestra

    data, sostituendo le approssimazioni aritmetiche con il vero CAGR geometrico.
    """
    T = len(r_replica)
    if T == 0:
        return ["-"] * 8

    # Performance assolute (CAGR geometrico vero)
    ricchezza_cumulata = (1.0 + r_replica).cumprod()
    cagr = (ricchezza_cumulata.iloc[-1] / ricchezza_cumulata.iloc[0]) ** (
        freq / T
    ) - 1.0
    volatilita_annua = r_replica.std() * np.sqrt(freq)
    indice_sharpe = cagr / volatilita_annua if volatilita_annua != 0 else 0.0
    max_drawdown = (
        1.0 - ricchezza_cumulata / ricchezza_cumulata.cummax()
    ).max()

    if e_target:
        return [
            f"{cagr*100:.2f}%",
            f"{volatilita_annua*100:.2f}%",
            f"{indice_sharpe:.2f}",
            f"{max_drawdown*100:.2f}%",
            "-",
            "-",
            "1.0000",
            "1.00",
        ]

    # Performance relative vs Benchmark Target
    rendimenti_attivi = r_replica - r_target
    tracking_error = rendimenti_attivi.std() * np.sqrt(freq)
    information_ratio = (
        (rendimenti_attivi.mean() * freq) / tracking_error
        if tracking_error != 0
        else 0.0
    )
    correlazione = np.corrcoef(r_replica, r_target)[0, 1]

    # Esposizione Lorda Media (Gross Exposure)
    if pesi_asset is None:
        esposizione_lorda = 1.0
    elif isinstance(pesi_asset, (int, float)):
        esposizione_lorda = float(pesi_asset)
    elif isinstance(pesi_asset, pd.Series):
        esposizione_lorda = pesi_asset.abs().sum()
    else:  # DataFrame di pesi dinamici
        esposizione_lorda = pesi_asset.abs().sum(axis=1).mean()

    return [
        f"{cagr*100:.2f}%",
        f"{volatilita_annua*100:.2f}%",
        f"{indice_sharpe:.2f}",
        f"{max_drawdown*100:.2f}%",
        f"{tracking_error*100:.2f}%",
        f"{information_ratio:.2f}",
        f"{correlazione:.4f}",
        f"{esposizione_lorda:.2f}",
    ]


etichette_metriche = [
    "1. CAGR Geometrico",
    "2. Volatilità Annualizzata",
    "3. Indice di Sharpe (Rf=0)",
    "4. Maximum Drawdown",
    "5. Tracking Error vs Target",
    "6. Information Ratio",
    "7. Correlazione con Target",
    "8. Esposizione Lorda Media",
]

# Costruzione Tabella Vista per Periodo (Confronto Side-by-Side)
dati_per_periodo = {
    ("1. Training Set (In-Sample)", "Target Index"): calcola_metriche_istituzionali(Y_train, Y_train, e_target=True),
    ("1. Training Set (In-Sample)", "OLS Statica"): calcola_metriche_istituzionali(Y_train, rendimenti_ols_train, pesi_ols_statici),
    ("1. Training Set (In-Sample)", "Kalman Dinamico"): calcola_metriche_istituzionali(Y_train, rendimenti_kf_train, pesi_kf_train),
    ("2. Validation Set (OOS 1)", "Target Index"): calcola_metriche_istituzionali(Y_val, Y_val, e_target=True),
    ("2. Validation Set (OOS 1)", "OLS Statica"): calcola_metriche_istituzionali(Y_val, rendimenti_ols_val, pesi_ols_statici),
    ("2. Validation Set (OOS 1)", "Kalman Dinamico"): calcola_metriche_istituzionali(Y_val, rendimenti_kf_val, pesi_kf_val),
    ("3. Test Set (OOS Puro 2)", "Target Index"): calcola_metriche_istituzionali(Y_test, Y_test, e_target=True),
    ("3. Test Set (OOS Puro 2)", "OLS Statica"): calcola_metriche_istituzionali(Y_test, rendimenti_ols_test, pesi_ols_statici),
    ("3. Test Set (OOS Puro 2)", "Kalman Dinamico"): calcola_metriche_istituzionali(Y_test, rendimenti_kf_test, pesi_kf_test),
}
tabella_master_periodi = pd.DataFrame(
    dati_per_periodo, index=etichette_metriche
)

# Costruzione Tabella Vista per Modello (Evoluzione Temporale Strategie)
dati_per_modello = {
    ("1. Benchmark Target", "Train"): calcola_metriche_istituzionali(
        Y_train, Y_train, e_target=True
    ),
    ("1. Benchmark Target", "Validation (OOS 1)"): (
        calcola_metriche_istituzionali(Y_val, Y_val, e_target=True)
    ),
    ("1. Benchmark Target", "Test (OOS 2)"): calcola_metriche_istituzionali(
        Y_test, Y_test, e_target=True
    ),
    ("2. OLS Statica Baseline", "Train"): calcola_metriche_istituzionali(
        Y_train, rendimenti_ols_train, pesi_ols_statici
    ),
    ("2. OLS Statica Baseline", "Validation (OOS 1)"): (
        calcola_metriche_istituzionali(
            Y_val, rendimenti_ols_val, pesi_ols_statici
        )
    ),
    ("2. OLS Statica Baseline", "Test (OOS 2)"): (
        calcola_metriche_istituzionali(
            Y_test, rendimenti_ols_test, pesi_ols_statici
        )
    ),
    ("3. Kalman Dinamico", "Train"): calcola_metriche_istituzionali(
        Y_train, rendimenti_kf_train, pesi_kf_train
    ),
    ("3. Kalman Dinamico", "Validation (OOS 1)"): (
        calcola_metriche_istituzionali(
            Y_val, rendimenti_kf_val, pesi_kf_val
        )
    ),
    ("3. Kalman Dinamico", "Test (OOS 2)"): calcola_metriche_istituzionali(
        Y_test, rendimenti_kf_test, pesi_kf_test
    ),
}
tabella_master_modelli = pd.DataFrame(
    dati_per_modello, index=etichette_metriche
)

print("=" * 110)
print(
    "TABELLA RIASSUNTIVA 1: CONFRONTO DI PRESA DIREATTA PER BLOCCO TEMPORALE"
)
print("=" * 110)
display(tabella_master_periodi)

print("\n" + "=" * 110)
print("TABELLA RIASSUNTIVA 2: DEGRADAZIONE STRUTTURALE PER MODELLO")
print("=" * 110)
display(tabella_master_modelli)

# %%
# ==============================================================================
# MODULO 7: RAPPRESENTAZIONE GRAFICA ISTITUZIONALE A 4 PANNELLI
# ==============================================================================
ricchezza_target = (1.0 + Y).cumprod()
ricchezza_ols = (1.0 + rendimenti_ols_totali).cumprod()
ricchezza_kf = (1.0 + rendimenti_kalman_netti).cumprod()

rend_attivi_ols = rendimenti_ols_totali - Y
rend_attivi_kf = rendimenti_kalman_netti - Y

te_mobile_ols = rend_attivi_ols.rolling(window=52).std() * np.sqrt(52)
te_mobile_kf = rend_attivi_kf.rolling(window=52).std() * np.sqrt(52)

fig, assi = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

# Pannello 1: Traiettoria di Ricchezza Cumulata
assi[0].plot(
    ricchezza_target.index,
    ricchezza_target,
    label="Indice Target (Benchmark)",
    color="darkblue",
    linewidth=2.2,
)
assi[0].plot(
    ricchezza_ols.index,
    ricchezza_ols,
    label="Baseline OLS Statica",
    color="crimson",
    linewidth=1.5,
    linestyle="--",
)
assi[0].plot(
    ricchezza_kf.index,
    ricchezza_kf,
    label="Replica Kalman Dinamico (Netta)",
    color="darkgreen",
    linewidth=1.8,
)
assi[0].axvline(
    Y_val.index[0],
    color="black",
    linestyle=":",
    linewidth=2,
    label="Split Val/Test (OOS)",
)
assi[0].axvline(Y_test.index[0], color="green", linestyle=":", linewidth=2)
assi[0].set_title(
    "1. Ricchezza Cumulata: Indice Target vs Strategie di Replica"
    " (Invarianza Causale)",
    fontsize=14,
    fontweight="bold",
)
assi[0].set_ylabel("Capitale (Base 1)")
assi[0].legend(loc="upper left", frameon=True)

# Pannello 2: Profilo del Maximum Drawdown
dd_target = 1.0 - ricchezza_target / ricchezza_target.cummax()
dd_ols = 1.0 - ricchezza_ols / ricchezza_ols.cummax()
dd_kf = 1.0 - ricchezza_kf / ricchezza_kf.cummax()

assi[1].plot(
    dd_target.index,
    dd_target * 100,
    label="Drawdown Target",
    color="darkblue",
    linewidth=1.5,
)
assi[1].plot(
    dd_ols.index,
    dd_ols * 100,
    label="Drawdown OLS",
    color="crimson",
    linewidth=1.2,
    linestyle="--",
)
assi[1].plot(
    dd_kf.index,
    dd_kf * 100,
    label="Drawdown Kalman Netto",
    color="darkgreen",
    linewidth=1.5,
)
assi[1].axvline(Y_val.index[0], color="black", linestyle=":", linewidth=2)
assi[1].axvline(Y_test.index[0], color="green", linestyle=":", linewidth=2)
assi[1].set_title(
    "2. Profilo di Drawdown Peak-to-Valley (%)",
    fontsize=14,
    fontweight="bold",
)
assi[1].set_ylabel("Drawdown (%)")
assi[1].legend(loc="upper left", frameon=True)

# Pannello 3: Rendimenti Attivi (Residui di Replica e_t)
assi[2].plot(
    rend_attivi_kf.index,
    rend_attivi_kf * 100,
    color="purple",
    linewidth=1.0,
    alpha=0.75,
    label="Active Return Settimanale Kalman ($e_t$ netto)",
)
assi[2].axhline(0, color="black", linestyle="-", linewidth=1.2)
assi[2].axvline(Y_val.index[0], color="black", linestyle=":", linewidth=2)
assi[2].axvline(Y_test.index[0], color="green", linestyle=":", linewidth=2)
assi[2].set_title(
    "3. Rendimenti Attivi Settimanali (Residuo di Replica: Kalman - Target)",
    fontsize=14,
    fontweight="bold",
)
assi[2].set_ylabel("Active Return (%)")
assi[2].legend(loc="upper left", frameon=True)

# Pannello 4: Tracking Error Mobile Annualizzato (Finestra 52 Settimane)
assi[3].plot(
    te_mobile_ols.index,
    te_mobile_ols * 100,
    label="Rolling TE 1 Anno - OLS Statica",
    color="crimson",
    linewidth=1.5,
    linestyle="--",
)
assi[3].plot(
    te_mobile_kf.index,
    te_mobile_kf * 100,
    label="Rolling TE 1 Anno - Kalman Dinamico (Netto)",
    color="darkred",
    linewidth=2.0,
)
assi[3].axvline(Y_val.index[0], color="black", linestyle=":", linewidth=2)
assi[3].axvline(Y_test.index[0], color="green", linestyle=":", linewidth=2)
assi[3].set_title(
    "4. Tracking Error Mobile Annualizzato (Finestra Rolling 52 Settimane)",
    fontsize=14,
    fontweight="bold",
)
assi[3].set_xlabel("Data di Rilevazione", fontsize=12)
assi[3].set_ylabel("Tracking Error (%)")
assi[3].legend(loc="upper left", frameon=True)

plt.tight_layout()
plt.show()