import streamlit as st
import pandas as pd
import numpy as np
from collections import deque

np.random.seed(42)

# ===================PAGE CONFIG=========================
st.set_page_config(
    page_title="AI Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ===================CSS===================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Main background */
.stApp { background: #0a0a0f; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12121f 100%);
    border-right: 1px solid #2a2a3e;
}
section[data-testid="stSidebar"] * { color: #c8c8e0 !important; }

/* Title */
.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #e8c97e, #f5a623, #e8c97e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1;
}
.sub-title {
    font-size: 0.85rem;
    color: #6060a0;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Pipeline step badges */
.pipeline-badge {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #3a3a5e;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: #8080c0;
    letter-spacing: 1px;
    margin: 3px;
}
.pipeline-badge.active {
    background: linear-gradient(135deg, #1a1a3e, #2a2a5e);
    border-color: #e8c97e;
    color: #e8c97e;
}

/* Movie card */
.movie-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    position: relative;
    transition: all 0.2s ease;
}
.movie-card:hover { border-color: #e8c97e; transform: translateY(-2px); }
.movie-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    color: #2a2a4e;
    position: absolute;
    top: 10px; right: 20px;
    line-height: 1;
}
.movie-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    color: #e8c97e;
    letter-spacing: 2px;
    margin-bottom: 2px;
}
.movie-meta {
    font-size: 0.78rem;
    color: #6060a0;
    margin-bottom: 12px;
}
.score-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0; }
.score-box {
    background: #0a0a18;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 8px 14px;
    text-align: center;
    min-width: 90px;
}
.score-label { font-size: 0.65rem; color: #5050a0; text-transform: uppercase; letter-spacing: 1px; }
.score-value { font-size: 1.2rem; font-weight: 600; color: #e8c97e; }
.score-value.good { color: #7ef5a0; }
.reason-box {
    background: #0d0d1e;
    border-left: 3px solid #e8c97e;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #a0a0c8;
    margin-top: 12px;
}

/* Log box */
.log-box {
    background: #080810;
    border: 1px solid #1e1e36;
    border-radius: 8px;
    padding: 14px 18px;
    font-family: 'Courier New', monospace;
    font-size: 0.76rem;
    color: #50d080;
    max-height: 280px;
    overflow-y: auto;
}
.log-step { color: #e8c97e; }
.log-ok   { color: #50d080; }
.log-info { color: #8080c0; }

/* Metrics */
.metric-pill {
    background: #1a1a2e;
    border: 1px solid #3a3a5e;
    border-radius: 50px;
    padding: 6px 18px;
    display: inline-block;
    margin: 4px;
    font-size: 0.8rem;
    color: #c0c0e8;
}
.metric-pill span { color: #e8c97e; font-weight: 600; }

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #c8a040, #e8c97e) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 32px !important;
    width: 100%;
    transition: all 0.2s !important;
}
div.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(232,201,126,0.3) !important; }

/* Sliders & selects */
div[data-baseweb="select"] > div { background: #12121f !important; border-color: #2a2a4e !important; color: #e8e8f0 !important; }
.stSlider > div { color: #e8e8f0 !important; }
div[data-testid="stMarkdownContainer"] h3 { color: #e8c97e !important; font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

/* Divider */
hr { border-color: #2a2a4e !important; }

/* Info boxes */
.stInfo { background: #0d0d22 !important; border-color: #3a3a6e !important; }
</style>
""", unsafe_allow_html=True)


# ===================AI BACKEND FUNCTIONS==========================

def parse_duration(d):
    d = str(d).strip()
    h, m = 0, 0
    if 'h' in d:
        parts = d.split('h')
        h = int(parts[0].strip())
        if 'm' in parts[1]:
            m = int(parts[1].replace('m','').strip())
    elif 'm' in d:
        m = int(d.replace('m','').strip())
    return h * 60 + m

@st.cache_data
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    df['duration_mins']  = df['duration'].apply(parse_duration)
    df['primary_genre']  = df['genres'].apply(lambda g: str(g).split(',')[0].strip())
    df = df.dropna(subset=['year','rating'])
    df['year']   = df['year'].astype(int)
    df['rating'] = df['rating'].astype(float)
    movies = []
    for _, row in df.iterrows():
        movies.append({
            "title"      : row['title'],
            "genre"      : row['primary_genre'],
            "year"       : row['year'],
            "duration"   : row['duration_mins'],
            "rating"     : row['rating'],
            "description": row.get('description',''),
        })
    return movies

# ==============================CSP Filters============================
def csp_filter(movies, genre, min_year, max_duration, min_rating):
    """CSP: keep movies passing ALL constraints."""
    valid = []
    for m in movies:
        if (m["genre"].lower() == genre.lower() and
            m["year"]     >= min_year and
            m["duration"] <= max_duration and
            m["rating"]   >= min_rating):
            valid.append(m)
    return valid

# ==============================BFS====================================
def bfs_search(movies):
    q, visited = deque(movies), []
    while q: visited.append(q.popleft())
    return visited
# =============================DFS====================================
def dfs_search(movies):
    s, visited = list(movies), []
    while s: visited.append(s.pop())
    return visited

# ==============================A*===================================
def heuristic(m): return round(10 - m["rating"], 2)

def a_star_search(movies):
    return sorted(movies, key=heuristic)

def normalise(vals):
    mn, mx = min(vals), max(vals)
    if mx == mn: return [0.5]*len(vals)
    return [(v-mn)/(mx-mn) for v in vals]

def euclidean(a, b):
    return float(np.sqrt(sum((x-y)**2 for x,y in zip(a,b))))

# ===========================K-MEAN==================================
def kmeans_cluster(movies, k=3):
    if len(movies) < k: return movies, k
    years     = normalise([m["year"]     for m in movies])
    durations = normalise([m["duration"] for m in movies])
    ratings   = normalise([m["rating"]   for m in movies])
    features  = [[years[i], durations[i], ratings[i]] for i in range(len(movies))]
    idx       = np.random.choice(len(features), k, replace=False)
    centroids = [features[i] for i in idx]
    assignments = [0]*len(movies)
    for _ in range(20):
        new_asgn = [min(range(k), key=lambda c: euclidean(f, centroids[c])) for f in features]
        if new_asgn == assignments: break
        assignments = new_asgn
        for cid in range(k):
            mems = [features[i] for i,a in enumerate(assignments) if a==cid]
            if mems: centroids[cid] = [sum(m[j] for m in mems)/len(mems) for j in range(3)]
    for i,m in enumerate(movies): m["cluster"] = assignments[i]
    best_cluster = movies[0]["cluster"]
    similar = [m for m in movies if m["cluster"] == best_cluster]
    return similar, k

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))

# =================================ANN=========================================
class TinyANN:
    def __init__(self):
        self.W1 = np.random.randn(3,4)*0.5
        self.b1 = np.random.randn(4)*0.1
        self.W2 = np.random.randn(4,1)*0.5
        self.b2 = np.random.randn(1)*0.1
    def predict(self, x):
        h = relu(x @ self.W1 + self.b1)
        o = sigmoid(h @ self.W2 + self.b2)
        return float(o[0])*10

def ann_predict(movies):
    if not movies: return movies
    net = TinyANN()
    years     = normalise([m["year"]     for m in movies])
    durations = normalise([m["duration"] for m in movies])
    ratings   = normalise([m["rating"]   for m in movies])
    for i, m in enumerate(movies):
        feat = np.array([years[i], durations[i], ratings[i]])
        m["predicted_rating"] = round(net.predict(feat), 2)
    return movies

# ====================================MINIMAX==================================
def minimax_score(movie):
    
    user_score = movie.get("predicted_rating", 0)
    general_score = movie.get("rating", 0)

    return min(user_score, general_score)

def apply_minimax(movies):
    
    for m in movies:
        m["minimax_score"] = round(minimax_score(m), 2)
    return movies


def rank_movies(movies, top_n):
    for m in movies:
        m["match_score"] = round(
            (m["rating"] + m["predicted_rating"] + m["minimax_score"]) / 3, 2
        )
    return sorted(movies, key=lambda m: m["match_score"], reverse=True)[:top_n]

def explain(movie, prefs):
    r = []
    r.append(f"genre matches your preference ({movie['genre']})")
    r.append(f"released in {movie['year']} (your min: {prefs['min_year']})")
    r.append(f"duration is {movie['duration']} min (your max: {prefs['max_duration']} min)")
    if movie["rating"] >= 8.5: r.append(f"IMDb rating {movie['rating']}/10 — excellent")
    elif movie["rating"] >= 8.0: r.append(f"IMDb rating {movie['rating']}/10 — great")
    else: r.append(f"IMDb rating {movie['rating']}/10 — good")
    return " • ".join(r)

# ==============================USER INPUT=================================
with st.sidebar:
    st.markdown("###  YOUR PREFERENCES")
    st.markdown("---")

    try:
        all_movies = load_dataset("imdb_top_movies.csv")
        available_genres = sorted(set(m["genre"] for m in all_movies))
    except:
        all_movies = []
        available_genres = ["Drama","Action","Crime","Comedy","Adventure","Sci-Fi","Horror","Thriller","Biography","Animation"]

    genre = st.selectbox(
        " Preferred Genre",
        options=available_genres,
        index=available_genres.index("Drama") if "Drama" in available_genres else 0
    )

    min_year = st.slider(
        " Released After (Year)",
        min_value=1920, max_value=2024,
        value=1990, step=1
    )

    max_duration = st.slider(
        "Max Duration (minutes)",
        min_value=60, max_value=400,
        value=180, step=5
    )

    min_rating = st.slider(
        " Minimum IMDb Rating",
        min_value=1.0, max_value=10.0,
        value=7.0, step=0.1
    )

    top_n = st.selectbox(
        " Number of Recommendations",
        options=[3, 5, 7, 10],
        index=1
    )

    st.markdown("---")
    run_btn = st.button("  GET RECOMMENDATIONS")

    st.markdown("---")

#=======================================INTERFACE==================================
st.markdown('<div class="main-title"> AI MOVIE RECOMMENDER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid AI Movie Recommendation & Decision System</div>', unsafe_allow_html=True)

# Pipeline steps
steps = ["CSP Filter","BFS Search","DFS Search","A* Search","K-Means","ANN Predict","Rank & Explain","Minimax"]
badges_html = "".join(f'<span class="pipeline-badge">{s}</span>' for s in steps)
st.markdown(badges_html, unsafe_allow_html=True)
st.markdown("---")

if not run_btn:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#12121f;border:1px solid #2a2a4e;border-radius:12px;padding:20px;text-align:center;'>
        <div style='font-size:2rem'></div>
        <div style='color:#e8c97e;font-family:Bebas Neue;font-size:1.1rem;letter-spacing:2px;margin:8px 0'>CSP FILTERING</div>
        <div style='color:#6060a0;font-size:0.8rem'>Hard constraints on genre, year, duration & rating filter irrelevant movies instantly</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#12121f;border:1px solid #2a2a4e;border-radius:12px;padding:20px;text-align:center;'>
        <div style='font-size:2rem'></div>
        <div style='color:#e8c97e;font-family:Bebas Neue;font-size:1.1rem;letter-spacing:2px;margin:8px 0'>AI SEARCH</div>
        <div style='color:#6060a0;font-size:0.8rem'>BFS, DFS & A* algorithms explore the movie space and surface the best candidates</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#12121f;border:1px solid #2a2a4e;border-radius:12px;padding:20px;text-align:center;'>
        <div style='font-size:2rem'></div>
        <div style='color:#e8c97e;font-family:Bebas Neue;font-size:1.1rem;letter-spacing:2px;margin:8px 0'>ML PREDICTION</div>
        <div style='color:#6060a0;font-size:0.8rem'>K-Means clusters similar movies, then ANN predicts a personalised rating for you</div>
        </div>""", unsafe_allow_html=True)


else:
    if not all_movies:
        st.error(" Could not load imdb_top_movies.csv")
        st.stop()

    prefs = {"genre": genre, "min_year": min_year, "max_duration": max_duration, "min_rating": min_rating}
    log   = []

    # CSP
    with st.spinner("Step 1 — CSP filtering movies..."):
        candidates = csp_filter(all_movies, genre, min_year, max_duration, min_rating)
        log.append(f'<span class="log-step">[CSP]</span>     Constraints applied: genre={genre}, year≥{min_year}, duration≤{max_duration}min, rating≥{min_rating}')
        log.append(f'<span class="log-ok">[CSP]</span>     {len(candidates)} / {len(all_movies)} movies passed all constraints')

    if not candidates:
        st.error(f"No movies matched your constraints. Try relaxing genre, year, duration, or rating.")
        genres_avail = sorted(set(m["genre"] for m in all_movies))
        st.info(f"Available genres: {', '.join(genres_avail)}")
        st.stop()

    # BFS 
    with st.spinner("Step 2 — BFS exploration..."):
        bfs_result = bfs_search(candidates)
        log.append(f'<span class="log-step">[BFS]</span>     Explored {len(bfs_result)} movies using Queue (FIFO)')

    # DFS 
    with st.spinner("Step 3 — DFS exploration..."):
        dfs_result = dfs_search(candidates)
        log.append(f'<span class="log-step">[DFS]</span>     Explored {len(dfs_result)} movies using Stack (LIFO)')

    # A*
    with st.spinner("Step 4 — A* search (best-first)..."):
        a_star_result = a_star_search(candidates)
        log.append(f'<span class="log-step">[A*]</span>      Best-first order by h(n)=10-rating')
        log.append(f'<span class="log-ok">[A*]</span>      Top candidate: {a_star_result[0]["title"]} (h={heuristic(a_star_result[0])})')

    # K-Means
    with st.spinner("Step 5 — K-Means clustering..."):
        k = min(3, len(a_star_result))
        clustered, k_used = kmeans_cluster(a_star_result, k=k)
        log.append(f'<span class="log-step">[K-Means]</span> Formed {k_used} clusters on year/duration/rating features')
        log.append(f'<span class="log-ok">[K-Means]</span> Best cluster contains {len(clustered)} similar movies')

    # ANN 
    with st.spinner("Step 6 — ANN predicting personalised ratings..."):
        with_preds = ann_predict(clustered)
        log.append(f'<span class="log-step">[ANN]</span>     Forward pass: ReLU hidden(4) Sigmoid output × 10')
        log.append(f'<span class="log-ok">[ANN]</span>     Predicted personalised ratings for {len(with_preds)} movies')

    # MINIMAX
    with st.spinner("Step 7 — Minimax decision (multi-user)..."):
       with_minimax = apply_minimax(with_preds)
       log.append(f'<span class="log-step">[MINIMAX]</span>  Resolving multi-user preferences')
       log.append(f'<span class="log-ok">[MINIMAX]</span>  Score = min(ANN, IMDb)')

    # Rank
    top_movies = rank_movies(with_minimax, top_n)
    log.append(f'<span class="log-step">[RANK]</span>    match_score = (IMDb + ANN) / 2  sorted descending')
    log.append(f'<span class="log-ok">[DONE]</span>    Top {len(top_movies)} recommendations ready ')

    # Pipeline Log + Stats 
    col_log, col_stats = st.columns([3, 2])

    with col_log:
        st.markdown("### AI Execution Steps")
        log_html = "<br>".join(f'<span class="log-info">$ </span>{l}' for l in log)
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    with col_stats:
        st.markdown("### Statistics")
        stats = [
            ("Dataset Size",    f"{len(all_movies)} movies"),
            ("After CSP",       f"{len(candidates)} movies"),
            ("After K-Means",   f"{len(clustered)} movies"),
            ("Recommended",     f"{len(top_movies)} movies"),
            ("Genre",           genre),
            ("Year Range",      f"{min_year} – 2024"),
            ("Max Duration",    f"{max_duration} min"),
            ("Min Rating",      f"{min_rating} / 10"),
        ]
        pills_html = "".join(
            f'<div class="metric-pill">{k}: <span>{v}</span></div>'
            for k, v in stats
        )
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("---")

    # ======================RESULTS==================================
    st.markdown(f"###  Top {len(top_movies)} Recommendations — {genre}")

    for rank, movie in enumerate(top_movies, 1):
        score_color = "good" if movie["match_score"] >= 8 else ""
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-rank">#{rank}</div>
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-meta">{movie['genre']} &nbsp;·&nbsp; {movie['year']} &nbsp;·&nbsp; {movie['duration']} min</div>
            <div class="score-row">
                <div class="score-box">
                    <div class="score-label">IMDb Rating</div>
                    <div class="score-value">{movie['rating']}<span style='font-size:0.7rem;color:#5050a0'>/10</span></div>
                </div>
                <div class="score-box">
                    <div class="score-label">ANN Prediction</div>
                    <div class="score-value">{movie['predicted_rating']}<span style='font-size:0.7rem;color:#5050a0'>/10</span></div>
                </div>
                <div class="score-box">
                    <div class="score-label">Match Score</div>
                    <div class="score-value {score_color}">{movie['match_score']}<span style='font-size:0.7rem;color:#5050a0'>/10</span></div>
                </div>
                <div class="score-box">
                    <div class="score-label">h(n) A*</div>
                    <div class="score-value" style='color:#a080ff'>{heuristic(movie)}</div>
                </div>
                <div class="score-box">
                    <div class="score-label">Minimax</div>
                    <div class="score-value" style='color:#ff8080'>{movie['minimax_score']}</div>
                </div>
            </div>
            <div class="reason-box">{explain(movie, prefs)}</div>
        </div>
        """, unsafe_allow_html=True)