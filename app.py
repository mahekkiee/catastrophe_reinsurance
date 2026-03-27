from flask import Flask, jsonify, send_file, request
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_FILE = DATA_DIR / 'irdai_claims_2014_2024.csv'

DATA_DIR.mkdir(exist_ok=True)

@app.route('/')
def index():
    return send_file(BASE_DIR / 'index.html')


# ================= DATA =================

def load_data():
    try:
        if not DATA_FILE.exists():
            return create_sample_data()
        return pd.read_csv(DATA_FILE)
    except:
        return create_sample_data()


def create_sample_data():
    data = {
        'Year': [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
        'NumClaims': [192500,735000,203500,211200,835000,217000,223500,895000,230800,237200,245000]
    }
    return pd.DataFrame(data)


# ================= API =================

@app.route('/api/trigger-analysis')
def trigger_analysis():
    selected_trigger = int(request.args.get('trigger', 800))

    triggers = [500,600,700,800,900,1000,1200]
    results = []

    for trigger in triggers:
        insurer_loss = 158.33
        premium = 0.03 * trigger
        total_cost = insurer_loss + premium

        results.append({
            'trigger': trigger,
            'insurer_loss': round(insurer_loss,2),
            'premium': round(premium,2),
            'total_cost': round(total_cost,2),
            'status': 'optimal' if trigger == selected_trigger else 'alt'
        })

    return jsonify({'success': True, 'data': results})


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
