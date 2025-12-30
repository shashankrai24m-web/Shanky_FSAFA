import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras import layers, models, losses
import random
import os

# ==========================================
# 1. CONFIGURATION & DETERMINISM
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Initializing NoMoreMehta.AI Forensic Engine...")

# ==========================================
# 2. SENSEX 200 DATABASE SIMULATION
# ==========================================
def generate_sensex_200():
    base_companies = [
        {'id': 'RELIANCE', 'name': 'Reliance Industries Ltd', 'industry': 'Energy / Utilities', 'revenue': 892932, 'profit': 69636, 'cfo': 141230, 'warranty': 2500},
        {'id': 'TCS', 'name': 'Tata Consultancy Services', 'industry': 'Technology', 'revenue': 240893, 'profit': 45908, 'cfo': 47900, 'warranty': 0},
        {'id': 'HDFCBANK', 'name': 'HDFC Bank Ltd', 'industry': 'Financial Services', 'revenue': 205000, 'profit': 46000, 'cfo': 50000, 'warranty': 0},
        {'id': 'ICICIBANK', 'name': 'ICICI Bank Ltd', 'industry': 'Financial Services', 'revenue': 123000, 'profit': 34000, 'cfo': 40000, 'warranty': 0},
        {'id': 'INFY', 'name': 'Infosys Limited', 'industry': 'Technology', 'revenue': 153670, 'profit': 26248, 'cfo': 24500, 'warranty': 0},
        {'id': 'HUL', 'name': 'Hindustan Unilever Ltd', 'industry': 'Retail / Consumer Goods', 'revenue': 59579, 'profit': 10143, 'cfo': 9800, 'warranty': 50},
        {'id': 'ITC', 'name': 'ITC Ltd', 'industry': 'Retail / Consumer Goods', 'revenue': 70000, 'profit': 19000, 'cfo': 18000, 'warranty': 0},
        {'id': 'SBI', 'name': 'State Bank of India', 'industry': 'Financial Services', 'revenue': 350000, 'profit': 50000, 'cfo': 100000, 'warranty': 0},
        {'id': 'BHARTIARTL', 'name': 'Bharti Airtel Ltd', 'industry': 'Technology', 'revenue': 139000, 'profit': 8300, 'cfo': 45000, 'warranty': 0},
        {'id': 'LENS', 'name': 'Lenskart Solutions', 'industry': 'Retail / Consumer Goods', 'revenue': 6652, 'profit': 297, 'cfo': 1336, 'warranty': 16.7},
        {'id': 'BAJFINANCE', 'name': 'Bajaj Finance', 'industry': 'Financial Services', 'revenue': 41400, 'profit': 11500, 'cfo': 4500, 'warranty': 0},
        {'id': 'LT', 'name': 'Larsen & Toubro', 'industry': 'Manufacturing', 'revenue': 183000, 'profit': 10500, 'cfo': 15000, 'warranty': 500},
        {'id': 'TATAMOTORS', 'name': 'Tata Motors', 'industry': 'Manufacturing', 'revenue': 345900, 'profit': 2400, 'cfo': 15000, 'warranty': 8500},
        {'id': 'AXISBANK', 'name': 'Axis Bank', 'industry': 'Financial Services', 'revenue': 106000, 'profit': 21000, 'cfo': 25000, 'warranty': 0},
        {'id': 'SUNPHARMA', 'name': 'Sun Pharmaceutical', 'industry': 'Pharmaceuticals', 'revenue': 43200, 'profit': 8500, 'cfo': 7800, 'warranty': 0},
        {'id': 'MARUTI', 'name': 'Maruti Suzuki', 'industry': 'Manufacturing', 'revenue': 112000, 'profit': 8000, 'cfo': 9500, 'warranty': 1200},
        {'id': 'ULTRACEMCO', 'name': 'UltraTech Cement', 'industry': 'Manufacturing', 'revenue': 63000, 'profit': 5000, 'cfo': 8000, 'warranty': 0},
        {'id': 'ASIANPAINT', 'name': 'Asian Paints', 'industry': 'Retail / Consumer Goods', 'revenue': 34480, 'profit': 4100, 'cfo': 3800, 'warranty': 45},
        {'id': 'TITAN', 'name': 'Titan Company', 'industry': 'Retail / Consumer Goods', 'revenue': 40575, 'profit': 3274, 'cfo': 1850, 'warranty': 120},
        {'id': 'HCLTECH', 'name': 'HCL Technologies', 'industry': 'Technology', 'revenue': 101450, 'profit': 14800, 'cfo': 16000, 'warranty': 0},
        {'id': 'WIPRO', 'name': 'Wipro Ltd', 'industry': 'Technology', 'revenue': 90480, 'profit': 11300, 'cfo': 11000, 'warranty': 0},
        {'id': 'NTPC', 'name': 'NTPC Ltd', 'industry': 'Energy / Utilities', 'revenue': 176000, 'profit': 17100, 'cfo': 21000, 'warranty': 0},
        {'id': 'M&M', 'name': 'Mahindra & Mahindra', 'industry': 'Manufacturing', 'revenue': 121000, 'profit': 10000, 'cfo': 8500, 'warranty': 1500},
        {'id': 'ADANIENT', 'name': 'Adani Enterprises', 'industry': 'Energy / Utilities', 'revenue': 136977, 'profit': 2472, 'cfo': 8100, 'warranty': 50},
        {'id': 'POWERGRID', 'name': 'Power Grid Corp', 'industry': 'Energy / Utilities', 'revenue': 45000, 'profit': 15000, 'cfo': 35000, 'warranty': 0},
        {'id': 'BAJAJFINSV', 'name': 'Bajaj Finserv', 'industry': 'Financial Services', 'revenue': 82000, 'profit': 6400, 'cfo': 12000, 'warranty': 0},
        {'id': 'NESTLEIND', 'name': 'Nestle India', 'industry': 'Retail / Consumer Goods', 'revenue': 19000, 'profit': 2800, 'cfo': 2500, 'warranty': 0},
        {'id': 'JSWSTEEL', 'name': 'JSW Steel', 'industry': 'Manufacturing', 'revenue': 165000, 'profit': 4100, 'cfo': 21000, 'warranty': 0},
        {'id': 'TATASTEEL', 'name': 'Tata Steel', 'industry': 'Manufacturing', 'revenue': 240000, 'profit': 8000, 'cfo': 25000, 'warranty': 0},
        {'id': 'LTIM', 'name': 'LTIMindtree', 'industry': 'Technology', 'revenue': 33180, 'profit': 4400, 'cfo': 4100, 'warranty': 0},
        {'id': 'GRASIM', 'name': 'Grasim Industries', 'industry': 'Manufacturing', 'revenue': 115000, 'profit': 6000, 'cfo': 12000, 'warranty': 0},
        {'id': 'ONGC', 'name': 'ONGC', 'industry': 'Energy / Utilities', 'revenue': 684000, 'profit': 38800, 'cfo': 55000, 'warranty': 1000},
        {'id': 'HINDALCO', 'name': 'Hindalco Industries', 'industry': 'Manufacturing', 'revenue': 220000, 'profit': 10000, 'cfo': 20000, 'warranty': 0},
        {'id': 'SBILIFE', 'name': 'SBI Life Insurance', 'industry': 'Financial Services', 'revenue': 80000, 'profit': 1800, 'cfo': 25000, 'warranty': 0},
        {'id': 'TECHM', 'name': 'Tech Mahindra', 'industry': 'Technology', 'revenue': 53290, 'profit': 4800, 'cfo': 5100, 'warranty': 0},
        {'id': 'BRITANNIA', 'name': 'Britannia Industries', 'industry': 'Retail / Consumer Goods', 'revenue': 16000, 'profit': 2000, 'cfo': 1800, 'warranty': 0},
        {'id': 'ADANIPORTS', 'name': 'Adani Ports', 'industry': 'Energy / Utilities', 'revenue': 20000, 'profit': 5000, 'cfo': 9000, 'warranty': 0},
        {'id': 'INDUSINDBK', 'name': 'IndusInd Bank', 'industry': 'Financial Services', 'revenue': 45000, 'profit': 8000, 'cfo': 9000, 'warranty': 0},
        {'id': 'CIPLA', 'name': 'Cipla Ltd', 'industry': 'Pharmaceuticals', 'revenue': 22000, 'profit': 2800, 'cfo': 3000, 'warranty': 0},
        {'id': 'EICHERMOT', 'name': 'Eicher Motors', 'industry': 'Manufacturing', 'revenue': 14000, 'profit': 3000, 'cfo': 2800, 'warranty': 100},
        {'id': 'DIVISLAB', 'name': 'Divis Laboratories', 'industry': 'Pharmaceuticals', 'revenue': 8000, 'profit': 2000, 'cfo': 1800, 'warranty': 0},
        {'id': 'DRREDDY', 'name': 'Dr. Reddys Labs', 'industry': 'Pharmaceuticals', 'revenue': 24000, 'profit': 4500, 'cfo': 4000, 'warranty': 0},
        {'id': 'BPCL', 'name': 'Bharat Petroleum', 'industry': 'Energy / Utilities', 'revenue': 420000, 'profit': 28000, 'cfo': 30000, 'warranty': 0},
        {'id': 'HEROMOTOCO', 'name': 'Hero MotoCorp', 'industry': 'Manufacturing', 'revenue': 34000, 'profit': 3000, 'cfo': 3500, 'warranty': 250},
        {'id': 'APOLLOHOSP', 'name': 'Apollo Hospitals', 'industry': 'Retail / Consumer Goods', 'revenue': 16000, 'profit': 800, 'cfo': 1200, 'warranty': 0},
        {'id': 'TATACONSUM', 'name': 'Tata Consumer', 'industry': 'Retail / Consumer Goods', 'revenue': 13000, 'profit': 1200, 'cfo': 1500, 'warranty': 0},
        {'id': 'COALINDIA', 'name': 'Coal India', 'industry': 'Energy / Utilities', 'revenue': 135000, 'profit': 28000, 'cfo': 35000, 'warranty': 0},
        {'id': 'BAJAJ_AUTO', 'name': 'Bajaj Auto', 'industry': 'Manufacturing', 'revenue': 36000, 'profit': 5600, 'cfo': 5000, 'warranty': 300},
        {'id': 'HDFCLIFE', 'name': 'HDFC Life Insurance', 'industry': 'Financial Services', 'revenue': 70000, 'profit': 1400, 'cfo': 18000, 'warranty': 0},
        {'id': 'UPL', 'name': 'UPL Ltd', 'industry': 'Manufacturing', 'revenue': 46000, 'profit': 3500, 'cfo': 6000, 'warranty': 0}
    ]
    
    sectors = ['Technology', 'Financial Services', 'Energy / Utilities', 'Retail / Consumer Goods', 'Manufacturing', 'Pharmaceuticals', 'Real Estate']
    extra_companies = []
    
    # Deterministic generation for the remaining ~150 to reach 200
    rng = np.random.RandomState(42)
    
    for i in range(150):
        sector = sectors[i % len(sectors)]
        base_rev = 5000 + rng.uniform() * 50000
        base_profit = base_rev * (rng.uniform() * 0.15 + 0.05)
        base_cfo = base_profit * (rng.uniform() * 0.4 + 0.8)
        base_warranty = base_rev * 0.01 if sector in ['Manufacturing', 'Retail / Consumer Goods'] else 0
        
        extra_companies.append({
            'id': f'SENSEX_{i+51}',
            'name': f'Sensex Company {i+51} ({sector})',
            'industry': sector,
            'revenue': int(base_rev),
            'profit': int(base_profit),
            'cfo': int(base_cfo),
            'warranty': int(base_warranty)
        })
        
    return base_companies + extra_companies

SCREENER_DB = generate_sensex_200()

# ==========================================
# 3. FORENSIC COMPONENT DEFINITIONS
# ==========================================
FEATURE_DESCRIPTIONS = {
    # --- Accruals & Cash Flow ---
    'Feat_0': {'name': 'Warranty Accrual', 'risk': 'High', 'desc': 'Estimated cost of future warranty claims. High in good years, low in bad years = "Cookie Jar".'},
    'Feat_1': {'name': 'Cash Paid for Warranties', 'risk': 'High', 'desc': 'Actual cash outflow. Low outflow with high accrual suggests fictitious reserves.'},
    'Feat_2': {'name': 'Net Income', 'risk': 'Medium', 'desc': 'Bottom line profit. Suspiciously smooth income over time is a red flag.'},
    'Feat_3': {'name': 'Deferred Revenue', 'risk': 'High', 'desc': 'Unearned revenue. Used to "sandbag" revenue for future periods.'},
    'Feat_4': {'name': 'Operating Cash Flow', 'risk': 'High', 'desc': 'Cash generated from core business. Should correlate with Net Income.'},
    'Feat_5': {'name': 'Change in Cash Sales', 'risk': 'High', 'desc': 'Growth in cash-collected revenue. Divergence from reported revenue suggests aggressive recognition.'},
    'Feat_7': {'name': 'Working Capital Accruals', 'risk': 'High', 'desc': 'Change in non-cash working capital. High values indicate earnings driven by estimates.'},
    'Feat_29': {'name': 'Change in Cash Margin', 'risk': 'High', 'desc': 'YoY change in CFO/Sales margin. Divergence from Net Income margin is a key warning.'},
    'Feat_34': {'name': 'Total Accruals', 'risk': 'Critical', 'desc': '(Net Income - CFO) / Assets. The "Sloan Anomaly" metric. High accruals predict lower returns.'},

    # --- Revenue & Receivables ---
    'Feat_6': {'name': 'Days Sales Receivables Index', 'risk': 'High', 'desc': 'Ratio of DSR in current vs prior year. Spike indicates booking revenue without cash collection.'},
    'Feat_22': {'name': 'Change in Receivables', 'risk': 'High', 'desc': 'Abnormal growth in AR. Often signals "Channel Stuffing".'},
    'Feat_18': {'name': 'Asset Turnover', 'risk': 'Medium', 'desc': 'Sales / Total Assets. Drop suggests inflated assets or fake sales.'},
    'Feat_28': {'name': 'Sales vs Inventory Growth', 'risk': 'Medium', 'desc': 'Inventory growing faster than sales signals demand issues or capitalized costs.'},

    # --- Inventory & Expenses ---
    'Feat_10': {'name': 'Inventory Reserve', 'risk': 'Medium', 'desc': 'Allowance for obsolete stock. Reducing this artificially boosts profit.'},
    'Feat_12': {'name': 'Gross Margin Index', 'risk': 'High', 'desc': 'Prior vs Current Gross Margin. Deteriorating margins create pressure to manipulate.'},
    'Feat_13': {'name': 'Change in Inventory', 'risk': 'High', 'desc': 'Inventory buildup. Rising inventory with falling sales is a warning sign.'},
    'Feat_15': {'name': 'SG&A Index', 'risk': 'Low', 'desc': 'SG&A/Sales ratio. Rising SGAI suggests loss of efficiency.'},
    'Feat_31': {'name': 'Depreciation Index', 'risk': 'Medium', 'desc': 'Slowing depreciation rate suggests extended asset lives to boost income.'},

    # --- Balance Sheet & Leverage ---
    'Feat_32': {'name': 'Working Capital to Assets', 'risk': 'Medium', 'desc': 'Liquidity measure. Sudden drop suggests distress.'},
    'Feat_35': {'name': 'Soft Assets Ratio', 'risk': 'Medium', 'desc': '% of assets that are intangible/goodwill. High values are prone to manipulation.'},
    'Feat_38': {'name': 'Asset Quality Index', 'risk': 'Medium', 'desc': 'Growth in non-current assets (excl PPE). Suggests cost capitalization.'},
    'Feat_40': {'name': 'Change in Long-Term Debt', 'risk': 'Medium', 'desc': 'Sudden debt spikes can indicate distress financing.'},
    'Feat_41': {'name': 'Debt to Assets', 'risk': 'High', 'desc': 'Leverage ratio. High leverage increases pressure to meet covenants.'}
}

# Add generic placeholders for 0-41 if not defined above
for i in range(42):
    key = f'Feat_{i}'
    if key not in FEATURE_DESCRIPTIONS:
        FEATURE_DESCRIPTIONS[key] = {'name': f'Forensic Feature {i}', 'risk': 'Low', 'desc': 'Standard forensic ratio.'}

# ==========================================
# 4. MANIFOLD LEARNING & STRUCTURAL ESTIMATION
# ==========================================

# Caches for deterministic models per industry
INDUSTRY_MODELS = {}
INDUSTRY_THRESHOLDS = {}

def get_or_train_model(industry, factor=1.0, safety_multiplier=0.8):
    """
    Returns a trained Autoencoder for the specific industry profile.
    Uses caching to ensure the 'Strict Limit' (Threshold) is mathematically constant.
    """
    if industry in INDUSTRY_MODELS:
        return INDUSTRY_MODELS[industry], INDUSTRY_THRESHOLDS[industry]
    
    print(f"Training new Manifold Model for Industry: {industry} (Factor: {factor})")
    
    # 1. Generate Synthetic 'Manifold' Data
    # Latent drivers (Demand, Efficiency, etc.)
    rng = np.random.RandomState(42) # Fixed seed for training data
    n_samples = 2000
    latent_dim = 5
    n_features = 42
    
    latent = rng.normal(0, 1, (n_samples, latent_dim))
    mixing = rng.normal(0, 1, (latent_dim, n_features))
    
    # Base Data = Latent * Mixing + Industry Noise
    X_raw = np.dot(latent, mixing)
    noise = rng.normal(0, 0.1 * factor, X_raw.shape)
    X_raw += noise
    
    # 2. Preprocess (RankGauss)
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train = scaler.fit_transform(X_raw)
    
    # 3. Build & Train Autoencoder
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(4, activation='relu', name='bottleneck'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(n_features, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=20, batch_size=256, verbose=0, shuffle=True)
    
    # 4. Calculate Deterministic Threshold
    reconstructions = model.predict(X_train)
    mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
    
    # 95th Percentile of Normal Noise
    base_limit = np.percentile(mse, 95)
    
    # Apply Conservative Safety Margin
    strict_limit = base_limit * safety_multiplier
    
    # Cache
    INDUSTRY_MODELS[industry] = (model, scaler) # Cache scaler too
    INDUSTRY_THRESHOLDS[industry] = strict_limit
    
    return (model, scaler), strict_limit

def get_industry_params(industry_name):
    """Returns volatility factor and safety multiplier based on sector risk."""
    lower = industry_name.lower()
    if 'retail' in lower or 'consumer' in lower:
        return 0.75, 0.80 # Stable, strict
    elif 'finance' in lower or 'bank' in lower:
        return 0.50, 0.70 # Regulated, very strict
    elif 'tech' in lower or 'software' in lower:
        return 1.20, 0.90 # Volatile, loose
    elif 'energy' in lower or 'utility' in lower:
        return 0.90, 0.85
    elif 'manufacturing' in lower or 'auto' in lower or 'pharma' in lower:
        return 0.85, 0.85
    else:
        return 1.0, 0.90 # General

def analyze_company(company_id):
    """
    Main entry point for the forensic analysis.
    """
    # 1. Fetch Data
    company = next((c for c in SCREENER_DB if c['id'] == company_id), None)
    if not company:
        return {"error": "Company not found in Sensex 200 Database"}
    
    # 2. Get Model
    factor, safety = get_industry_params(company['industry'])
    (model, scaler), threshold = get_or_train_model(company['industry'], factor, safety)
    
    # 3. Construct Forensic Vector (Log-Normalized Ratios)
    # We use a deterministic RNG seeded by company name to simulate the specific feature values
    # In a real app, these would be calculated from detailed line items.
    comp_seed = sum(ord(c) for c in company_id)
    rng = np.random.RandomState(comp_seed)
    
    vector = rng.uniform(-1, 1, 42) # Base noise
    
    # Inject Specific Financial Logic based on DB values
    rev = company['revenue']
    war = company['warranty']
    cfo = company['cfo']
    
    # Feat 0: Warranty Accrual Ratio
    if rev > 0:
        w_ratio = war / rev
        # Map to z-score space roughly: 0.2% -> -2, 5% -> +3
        vector[0] = np.log(w_ratio + 0.0001) + 6
        vector[0] = np.clip(vector[0], -5, 5)
        
    # Feat 1: Cash Flow Divergence
    if rev > 0:
        c_ratio = cfo / rev
        # Map 15% margin -> 0. 
        vector[1] = (c_ratio - 0.15) * 10
        vector[1] = np.clip(vector[1], -5, 5)
        
    # 4. Inference
    # Scale input
    input_vec = scaler.transform(vector.reshape(1, -1))
    
    # Predict
    reconstruction = model.predict(input_vec)
    reconstruction_error = np.mean(np.square(input_vec - reconstruction))
    
    # 5. Structural Estimation (Signal vs Noise)
    # Extract latent signal strength
    encoder = models.Model(inputs=model.input, outputs=model.get_layer('bottleneck').output)
    latent_vec = encoder.predict(input_vec)
    signal_strength = np.sum(np.square(latent_vec)) # Sigma_v
    
    noise_ratio = reconstruction_error / (signal_strength + 0.01)
    
    # 6. Feature Attribution
    diff = np.square(input_vec - reconstruction)[0]
    
    top_indices = diff.argsort()[::-1] # Sort desc
    components = []
    for idx in top_indices:
        key = f"Feat_{idx}"
        info = FEATURE_DESCRIPTIONS.get(key, {'name': key, 'risk': 'Low'})
        components.append({
            'id': key,
            'name': info['name'],
            'error': float(diff[idx]),
            'risk': info['risk'],
            'desc': info['desc']
        })
        
    is_anomaly = reconstruction_error > threshold
    
    return {
        "company": company,
        "analysis": {
            "score": float(reconstruction_error),
            "threshold": float(threshold),
            "noise_ratio": float(noise_ratio),
            "is_anomaly": bool(is_anomaly),
            "verdict": "COOKIE JAR DETECTED" if is_anomaly else "CLEAN",
            "safety_margin": f"{int((1-safety)*100)}%",
            "components": components
        }
    }

# ==========================================
# 5. EXECUTION BLOCK (Demo)
# ==========================================
if __name__ == "__main__":
    print("\n--- NoMoreMehta.AI: Automated Forensic Audit ---\n")
    
    # Example: Analyze a few companies
    targets = ['LENS', 'TCS', 'RELIANCE', 'HDFCBANK']
    
    for tid in targets:
        result = analyze_company(tid)
        print(f"Analyzing: {result['company']['name']} ({result['company']['industry']})")
        print(f"  > Strict Limit: {result['analysis']['threshold']:.4f} (Safety Margin: {result['analysis']['safety_margin']})")
        print(f"  > Anomaly Score: {result['analysis']['score']:.4f}")
        print(f"  > Noise Ratio: {result['analysis']['noise_ratio']:.2f}")
        print(f"  > Verdict: {result['analysis']['verdict']}")
        
        if result['analysis']['is_anomaly']:
            print("  > Top Risk Factors:")
            for comp in result['analysis']['components'][:3]:
                print(f"    - {comp['name']} (Error: {comp['error']:.2f}) [{comp['risk']}]")
        print("-" * 50)
