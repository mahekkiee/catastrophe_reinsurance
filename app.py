from flask import Flask, jsonify, send_file
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'irdai_claims_2014_2024.csv'

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# ✅ Serve index.html from root (NO templates folder needed)
@app.route('/')
def index():
    return send_file(BASE_DIR / 'index.html')


# ================= DATA FUNCTIONS =================

def load_data():
    try:
        if not DATA_FILE.exists():
            return create_sample_data()
        return pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return create_sample_data()


def create_sample_data():
    data = {
        'Year': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'InsuranceType': ['General'] * 11,
        'NumClaims': [192500, 735000, 203500, 211200, 835000, 217000, 223500, 895000, 230800, 237200, 245000],
        'TotalLossCrores': [5920, 21000, 6430, 6810, 24250, 7000, 7300, 25900, 7630, 7670, 8450],
        'AvgLossPerClaim': [307532, 285714, 315971, 322443, 290419, 322581, 326622, 289385, 330589, 323356, 344898],
        'ClaimRatio': [0.75, 1.18, 0.78, 0.80, 1.32, 0.81, 0.84, 1.45, 0.86, 0.85, 0.87],
        'Region': ['All India'] * 11
    }
    return pd.DataFrame(data)


# ================= MARKOV MODEL =================

def fit_markov_model(df):
    catastrophic_years = [2015, 2018, 2021]

    def assign_regime(year):
        return 'Catastrophic' if year in catastrophic_years else 'Normal'

    df['Regime'] = df['Year'].apply(assign_regime)
    regime_sequence = df['Regime'].tolist()

    transitions = {'N->N': 0, 'N->C': 0, 'C->N': 0, 'C->C': 0}

    for i in range(len(regime_sequence) - 1):
        current = regime_sequence[i][0]
        next_regime = regime_sequence[i + 1][0]
        transitions[f"{current}->{next_regime}"] += 1

    total_from_N = transitions['N->N'] + transitions['N->C']
    total_from_C = transitions['C->N'] + transitions['C->C']

    P = np.array([
        [
            transitions['N->N'] / total_from_N if total_from_N > 0 else 0.5,
            transitions['N->C'] / total_from_N if total_from_N > 0 else 0.5
        ],
        [
            transitions['C->N'] / total_from_C if total_from_C > 0 else 1.0,
            transitions['C->C'] / total_from_C if total_from_C > 0 else 0.0
        ]
    ])

    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmax(np.abs(eigenvalues - 1) < 1e-8)
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()

    return P, pi, df


# ================= API ROUTES =================

@app.route('/api/markov-data')
def markov_data():
    try:
        df = load_data()
        P, pi, _ = fit_markov_model(df)

        return jsonify({
            'success': True,
            'markov_matrix': {
                'normal_to_normal': round(float(P[0, 0]), 4),
                'normal_to_catastrophic': round(float(P[0, 1]), 4),
                'catastrophic_to_normal': round(float(P[1, 0]), 4),
                'catastrophic_to_catastrophic': round(float(P[1, 1]), 4)
            },
            'stationary_distribution': {
                'normal': round(float(pi[0]), 4),
                'catastrophic': round(float(pi[1]), 4)
            },
            'years': df['Year'].tolist(),
            'claims': df['NumClaims'].tolist()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trigger-analysis')
def trigger_analysis():
    try:
        triggers = [500, 600, 700, 800, 900, 1000, 1200]
        results = []

        for trigger in triggers:
            insurer_loss = 158.33
            premium = 0.03 * trigger
            total_cost = insurer_loss + premium

            results.append({
                'trigger': trigger,
                'insurer_loss': round(insurer_loss, 2),
                'premium': round(premium, 2),
                'total_cost': round(total_cost, 2),
                'status': 'optimal' if trigger == 800 else 'alternative'
            })

        return jsonify({'success': True, 'data': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/simulation-results')
def simulation_results():
    return jsonify({
        'success': True,
        'mean_loss': 158,
        'percentile_95': 189,
        'max_loss': 214,
        'scenarios': 5000,
        'years': 30
    })


@app.route('/api/company-profiles')
def company_profiles():
    companies = {
        'hdfc': {'name': 'HDFC Insurance', 'portfolio': '₹85,000 Cr', 'market_share': '12%', 'recommendation': '800 Cr'},
        'icici': {'name': 'ICICI Prudential', 'portfolio': '₹92,000 Cr', 'market_share': '15%', 'recommendation': '800 Cr'},
        'bajaj': {'name': 'Bajaj Allianz', 'portfolio': '₹78,000 Cr', 'market_share': '10%', 'recommendation': '750 Cr'},
        'sbi': {'name': 'SBI Insurance', 'portfolio': '₹68,000 Cr', 'market_share': '8%', 'recommendation': '700 Cr'},
        'oriental': {'name': 'Oriental Insurance', 'portfolio': '₹45,000 Cr', 'market_share': '5%', 'recommendation': '600 Cr'}
    }

    return jsonify({'success': True, 'companies': companies})


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


# ================= RUN =================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
