from flask import Flask, jsonify, send_file, request
import pandas as pd
import os
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent

@app.route('/')
def index():
    return send_file(BASE_DIR / 'index.html')


@app.route('/api/trigger-analysis')
def trigger_analysis():
    selected_trigger = int(request.args.get('trigger', 800))

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
            'status': 'optimal' if trigger == selected_trigger else 'alt'
        })

    return jsonify({'success': True, 'data': results})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
