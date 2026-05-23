#Capstone Project: Ebuss - Sentiment-Based Product Recommendation System
#Name: Suraj Bhadra, Cohort: C 75

from flask import Flask, render_template_string, request
from model import recommend_products, valid_usernames

app = Flask(__name__)


# HTML Template 

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ebuss - Product Recommendation System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 40px 20px;
        }
        .container {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            max-width: 700px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            text-align: center;
            color: #333;
            font-size: 28px;
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .form-group {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
            outline: none;
        }
        input[type="text"]:focus {
            border-color: #667eea;
        }
        button {
            padding: 12px 28px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102,126,234,0.4);
        }
        .results {
            margin-top: 24px;
        }
        .results h2 {
            color: #333;
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #667eea;
        }
        .product-card {
            background: #f8f9ff;
            border-left: 4px solid #667eea;
            padding: 14px 18px;
            margin-bottom: 10px;
            border-radius: 0 8px 8px 0;
            transition: background 0.2s;
        }
        .product-card:hover {
            background: #eef0ff;
        }
        .product-rank {
            font-weight: 700;
            color: #764ba2;
            margin-right: 10px;
        }
        .product-name {
            color: #333;
            font-size: 15px;
        }
        .error {
            background: #fff5f5;
            border: 1px solid #fc8181;
            color: #c53030;
            padding: 14px 18px;
            border-radius: 8px;
            margin-top: 16px;
        }
        .info {
            color: #888;
            font-size: 13px;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ebuss</h1>
        <p class="subtitle">Sentiment-Based Product Recommendation System</p>

        <form method="POST" action="/">
            <div class="form-group">
                <input type="text" name="username" id="username"
                       placeholder="Enter a username (e.g., {{ sample_user }})"
                       value="{{ username or '' }}" autocomplete="off">
                <button type="submit">Submit</button>
            </div>
        </form>

        {% if recommendations %}
        <div class="results">
            <h2>Top 5 Recommendations for "{{ username }}"</h2>
            {% for product in recommendations %}
            <div class="product-card">
                <span class="product-rank">#{{ loop.index }}</span>
                <span class="product-name">{{ product }}</span>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <p class="info">Enter an existing username from the dataset to get personalised product recommendations.</p>
    </div>
</body>
</html>
"""



# Routes

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    error = None
    username = None
    sample_user = valid_usernames[0] if valid_usernames else 'joshua'

    if request.method == 'POST':
        username = request.form.get('username', '').strip()

        if not username:
            error = "Please enter a username."
        elif username not in valid_usernames:
            error = (f"Username '{username}' not found in the dataset. "
                     f"Please enter a valid username.")
        else:
            recs = recommend_products(username)
            if recs:
                recommendations = recs
            else:
                error = f"Could not generate recommendations for '{username}'."

    return render_template_string(
        HTML_TEMPLATE,
        username=username,
        recommendations=recommendations,
        error=error,
        sample_user=sample_user
    )



# Run

if __name__ == '__main__':
    print("Starting Ebuss Recommendation System...")
    print(f"Total users available: {len(valid_usernames)}")
    print(f"Sample usernames: {valid_usernames[:5]}")
    app.run(debug=True, host='127.0.0.1', port=5000)
