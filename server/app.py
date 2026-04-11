import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import os
import sys
import copy
import json
from typing import Dict, Any, List, Tuple

# Ensure root directory is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import BusRoutingEnv, Observation, Action, Reward
from tasks import get_task, TASK_MEDIUM
from agent import DQNAgent
from sessions import store as session_store

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openai import OpenAI
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# API Configuration (from Environment Secrets)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
FREE_MODELS = [
    "openai/gpt-oss-120b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free"
]
HF_MODELS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]
MODEL_NAME = os.getenv("MODEL_NAME", FREE_MODELS[0])
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# Training Analytics Helpers
# ---------------------------------------------------------------------------

def load_training_metrics():
    """Load training convergence data from CSV if available."""
    paths = [
        "models/training_metrics_v6.csv",
        "models/training_metrics.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return None

def create_convergence_plots():
    """Generate training analytics plots from saved metrics."""
    df = load_training_metrics()
    if df is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No training metrics found. Run: python train.py",
            showarrow=False, font=dict(size=12, color="#94a3b8")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False), height=300
        )
        return fig

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "🏆 Episode Reward (Convergence)",
            "📉 Training Loss (Decay)",
            "🎲 Epsilon (Exploration Schedule)"
        ],
        horizontal_spacing=0.08,
    )

    # Reward curve with rolling average
    episodes = df["episode"].values
    rewards = df["total_reward"].values
    window = max(5, len(rewards) // 20)
    rolling = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, name="Raw Reward",
        line=dict(color="rgba(56,189,248,0.3)", width=1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=episodes, y=rolling, name="Smoothed",
        line=dict(color="#38bdf8", width=3),
    ), row=1, col=1)

    # Loss curve
    if "loss" in df.columns:
        loss = df["loss"].values
        loss_rolling = pd.Series(loss).rolling(window=window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=episodes, y=loss_rolling, name="Loss",
            line=dict(color="#f87171", width=2),
        ), row=1, col=2)

    # Epsilon schedule
    if "epsilon" in df.columns:
        fig.add_trace(go.Scatter(
            x=episodes, y=df["epsilon"].values, name="ε",
            line=dict(color="#a78bfa", width=2),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.1)',
        ), row=1, col=3)

    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#94a3b8", size=10),
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig

def create_error_fig(msg: str):
    """Helper to create a plotly figure displaying an error message."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error: {msg}",
        showarrow=False, font=dict(size=14, color="#f87171")
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=300
    )
    return fig

# ---------------------------------------------------------------------------
# Globals / State
# ---------------------------------------------------------------------------

MODELS_DIR = "models"
DEFAULT_MODEL = os.path.join(MODELS_DIR, "dqn_bus_v6_best.pt")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(MODELS_DIR, "dqn_bus_v5.pt")

class SessionState:
    def __init__(self):
        # Primary RL Agent
        self.env_rl = None
        self.agent = None
        self.obs_rl = None
        
        # Baseline Agent (Greedy)
        self.env_base = None
        self.obs_base = None
        
        self.done = False
        self.reward_history_rl = []
        self.reward_history_base = []
        
        self.last_q_values = np.zeros(3)
        self.last_reason = "System Initialized"
        self.compare_mode = True  # Enable by default for better demo
        self.difficulty = "medium"
        self.agent_mode = "Dueling DDQN (Local)"

class HeuristicAgent:
    """A rule-based agent that acts as a reliable fallback when the DQN model is missing."""
    def predict_q_values(self, obs: np.ndarray) -> np.ndarray:
        # obs = [pos, fuel, onboard, q0, q1, q2, time]
        q0, q1, q2 = obs[3], obs[4], obs[5]
        fuel = obs[1]
        
        q_vals = np.zeros(3)
        # Decision logic for visual feedback
        if fuel < 15:
            q_vals[2] = 10.0 # Prioritize waiting to save fuel
        elif q0 > 8:
            q_vals[2] = 15.0 # Wait if many people are here
        elif q1 > q0 + 5:
            q_vals[0] = 12.0 # Move to next if queue is much larger
        else:
            q_vals[0] = 5.0  # Default to move+pickup
        return q_vals

class LLMAgent:
    """Agent that queries OpenRouter/OpenAI for decisions."""
    SYSTEM_PROMPT = (
        "You are an Elite Global Transit Optimizer managing a metropolitan bus network. "
        "Your objective is to maximize total passenger pickups while minimizing fuel waste.\n\n"
        "OBS FORMAT: [bus_pos, fuel (0-100), onboard_pax, q_current, q_next, q_after_next, time_step]\n\n"
        "ACTIONS:\n"
        "  0 = MOVE + PICKUP (Standard operation)\n"
        "  1 = MOVE + SKIP   (Use to bypass low-demand stops or if bus is full)\n"
        "  2 = WAIT + PICKUP (Use to clear high-demand bottlenecks)\n\n"
        "STRATEGIC GUIDELINES:\n"
        "- If the next station (q_next) has much higher demand than current stop (q_current), consider skipping or moving quickly.\n"
        "- If fuel is < 20, prioritize WAITING (costs 0.2) over MOVING (costs 1.0) unless passenger demand is critical.\n"
        "- If bus is near capacity (30+), SKIP stops with low demand to reach terminal faster.\n\n"
        "Respond ONLY with a JSON object: {\"action\": <0,1,2>, \"reason\": \"<strategic reasoning>\"}"
    )

    def __init__(self):
        # OpenRouter requirements: site_url and app_name headers
        self.headers = {
            "HTTP-Referer": "https://huggingface.co/spaces",
            "X-Title": "OpenEnv Bus Optimizer"
        }
        self.client = OpenAI(
            base_url=API_BASE_URL, 
            api_key=OPENAI_API_KEY,
            default_headers=self.headers
        )
        self.model_list = FREE_MODELS
        # Ensure the user's preferred model is at the front
        if MODEL_NAME not in self.model_list:
            self.model_list = [MODEL_NAME] + self.model_list
            
        # Initialize HF Client
        self.hf_client = None
        if HF_TOKEN:
            self.hf_client = InferenceClient(token=HF_TOKEN)
            self.hf_models = HF_MODELS

    def predict_q_values(self, obs: np.ndarray) -> Tuple[np.ndarray, str]:
        # Since LLMs return actions, we mock Q-values for the UI (1.0 for chosen)
        user_msg = f"Observation: {obs.tolist()}. Choose action (0, 1, or 2)."
        
        last_err = ""
        for model in self.model_list:
            try:
                # Use streaming to capture reasoning tokens/usage
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                    temperature=0.0,
                    max_tokens=200,
                    stream=True,
                    stream_options={"include_usage": True},
                    timeout=10.0
                )
                
                full_text = ""
                reasoning_tokens = 0
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_text += chunk.choices[0].delta.content
                    if chunk.usage:
                        # Capture reasoning tokens if available (OpenAI schema)
                        reasoning_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                
                # Clean possible markdown
                text = full_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(text)
                act = int(data.get("action", 0))
                reason = data.get("reason", "Strategic alignment achieved.")
                
                # Mock Q-values (highest for chosen)
                q_vals = np.zeros(3)
                q_vals[act] = 10.0
                for i in range(3):
                    if i != act: q_vals[i] = 2.0
                
                # Get a pretty name for the model
                model_label = model.split("/")[-1].split(":")[0].upper()
                intelligence_badge = f"<span class='badge' style='background:rgba(139,92,246,0.1); color:#a78bfa; margin-left:10px; border:1px solid rgba(139,92,246,0.2)'>🧠 NEURAL LOAD: {reasoning_tokens}t</span>" if reasoning_tokens > 0 else ""
                
                return q_vals, f"<b style='color:#0ea5e9'>[AI: {model_label}]</b> {intelligence_badge} <br>{reason}"
            except Exception as e:
                # Capture the inner message if it's a 429/400 from OpenRouter
                err_text = str(e)
                if hasattr(e, 'response'):
                    try: err_text = e.response.json().get('error', {}).get('message', str(e))
                    except: pass
                
                last_err = err_text
                print(f"Model {model} failed: {err_text}")
                continue # Try the next model
        
        # --- SECONDARY FALLBACK: Hugging Face Inference API ---
        if self.hf_client:
            for hf_model in self.hf_models:
                try:
                    # HF Inference Client uses a slightly different API
                    response = self.hf_client.chat_completion(
                        model=hf_model,
                        messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
                        max_tokens=60,
                        temperature=0.01
                    )
                    text = response.choices[0].message.content.strip()
                    text = text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(text)
                    act = int(data.get("action", 0))
                    reason = data.get("reason", "Secondary HF Strategy applied.")
                    
                    q_vals = np.zeros(3)
                    q_vals[act] = 10.0
                    for i in range(3):
                        if i != act: q_vals[i] = 2.0
                    
                    return q_vals, f"<b style='color:#a78bfa'>[AI: HF-{hf_model.split('/')[-1].upper()}]</b> {reason}"
                except Exception as hf_e:
                    print(f"HF Model {hf_model} failed: {hf_e}")
                    continue
        
        # All models failed (Fallback to heuristic)
        h = HeuristicAgent()
        return h.predict_q_values(obs), f"<b style='color:#f87171'>[OFFLINE FALLBACK]</b> All online models failed. Using backup heuristic. Error: {last_err[:40]}..."

def test_api_key():
    """Simple ping to OpenRouter to verify connectivity and API key."""
    if not OPENAI_API_KEY:
        return "<span class='badge badge-blue' style='background:#f87171; color:white;'>❌ NO KEY PROVIDED</span>"
    try:
        client = OpenAI(
            base_url=API_BASE_URL, 
            api_key=OPENAI_API_KEY,
            default_headers={
                "HTTP-Referer": "https://huggingface.co/spaces",
                "X-Title": "OpenEnv Bus Optimizer Test"
            }
        )
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        return "<span class='badge badge-green'>✅ API KEY ACTIVE (CONNECTED)</span>"
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response'):
            try:
                # Try to extract the specific OpenRouter error message
                error_msg = e.response.json().get('error', {}).get('message', str(e))
            except: pass
        return f"<span class='badge' style='background:#f87171; color:white;'>❌ OpenRouter Error: {error_msg}</span>"

state = SessionState()

# --- OpenEnv API Implementation (for Automated Validators) ---
api_app = FastAPI(title="OpenEnv Bus RL API")
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared background environment for API calls
api_env = TASK_MEDIUM.build_env()

@api_app.post("/reset")
async def api_reset(req: Dict[str, str] = Body(default={})):
    """
    OpenEnv standard reset endpoint.
    Optionally accepts task_id to start a specific scenario.
    Returns observation and a session_id for future steps.
    """
    task_id = req.get("task_id", "task2_1")
    # Support both episode_id (for tracking) and session_id (for state)
    session_id = req.get("session_id", req.get("episode_id"))
    
    if not session_id:
        # Create a new session if none provided
        from sessions import store as s_store
        session_id = s_store.create_session(task_id)
        env = s_store.get_env(session_id)
    else:
        # Use existing session if valid
        from sessions import store as s_store
        env = s_store.get_env(session_id)
        if not env:
            session_id = s_store.create_session(task_id)
            env = s_store.get_env(session_id)
            
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "session_id": session_id,
        "episode_id": session_id # for compatibility
    }

@api_app.post("/step")
async def api_step(action_req: Dict[str, Any] = Body(...)):
    """
    OpenEnv standard step endpoint.
    Requires session_id and action.
    """
    session_id = action_req.get("session_id", action_req.get("episode_id"))
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id or episode_id required for /step")
        
    from sessions import store as s_store
    env = s_store.get_env(session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired")
        
    act_val = action_req.get("action", 0)
    obs, reward, done, info = env.step(act_val)
    
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": bool(done),
        "info": info,
        "session_id": session_id
    }

@api_app.get("/state")
async def api_state():
    """OpenEnv standard state endpoint."""
    return api_env.state()

@api_app.get("/tasks")
async def api_tasks():
    """List available tasks and their configurations."""
    from tasks import TASKS
    return {k: v.to_dict() for k, v in TASKS.items()}

@api_app.post("/grader")
async def api_grader(req: Dict[str, Any] = Body(...)):
    """
    OpenEnv standard grader endpoint.
    Expects JSON body with "task_id" and "action".
    """
    import grader
    
    task_id = req.get("task_id", "task1_1")
    
    # If the request wants to grade a specific task with a given action
    if "action" in req:
        action = req["action"]
        session_id = req.get("session_id", req.get("episode_id"))
        
        if session_id:
            from sessions import store as s_store
            env = s_store.get_env(session_id)
            if not env:
                session_id = s_store.create_session(task_id)
                env = s_store.get_env(session_id)
        else:
            env = api_env
            
        obs, reward, done, info = env.step(action)
        # Normalize reward to (0, 1) range strictly
        score = float(np.clip((reward.value + 10) / 20.0, 0.05, 0.95))
        return {
            "task_id": task_id,
            "score": score,
            "reward": reward.value,
            "done": bool(done),
            "session_id": session_id
        }
    
    # Full task grade
    func_name = f"grade_{task_id.replace('-', '_')}"
    if hasattr(grader, func_name):
        from agent import DQNAgent
        agent = DQNAgent.load(DEFAULT_MODEL)
        policy = lambda obs: agent.act(obs, greedy=True)
        
        grade_func = getattr(grader, func_name)
        score = grade_func(policy, episodes=2)
        return {
            "task_id": task_id,
            "score": float(np.clip(score, 0.05, 0.95)),
            "status": "completed"
        }
        
    raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

@api_app.get("/baseline")
async def api_baseline():
    """Return pre-computed baseline scores."""
    return {
        "task1_1": 0.50,
        "task2_1": 0.48,
        "task3_1": 0.45,
        "description": "Baseline performance of a simple greedy heuristic on primary variants."
    }

@api_app.get("/health")
async def health():
    return {"status": "healthy", "env": "rl-bus-optimization"}

# --- Gradio UI Mapping ---
ACTION_MAP = {
    0: "MOVE + PICKUP",
    1: "MOVE + SKIP",
    2: "WAIT + PICKUP",
}

# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def create_comparison_plot(render_rl: Dict[str, Any], render_base: Dict[str, Any] = None):
    """Creates a high-end bus route map with Apple-style aesthetics."""
    stops = render_rl["stops"]
    fig = go.Figure()
    
    # Path with subtle glow
    fig.add_trace(go.Scatter(
        x=[-0.5, len(stops)-0.5], y=[0]*2,
        mode='lines', line=dict(color='rgba(255,255,255,0.05)', width=8),
        hoverinfo='none', showlegend=False
    ))
    
    # Stops with high-end tooltips
    fig.add_trace(go.Scatter(
        x=[s["stop_idx"] for s in stops], y=[0] * len(stops),
        mode='markers', name='Stations',
        marker=dict(size=12, color='rgba(255,255,255,0.4)', symbol='circle-open', line=dict(width=2)),
        hoverinfo='text',
        text=[f"Station {s['stop_idx']} | Queue: {int(s['queue_len'])}" for s in stops]
    ))

    # Real-time Queues (Gradients)
    fig.add_trace(go.Bar(
        x=[s["stop_idx"] for s in stops], y=[s["queue_len"] for s in stops],
        marker=dict(color='#0ea5e9', opacity=0.3),
        name="Station Demand", hoverinfo='skip'
    ))

    # Bus Markers (Stellar Blue for RL, Ghostly Gray for Baseline)
    if render_base:
        fig.add_trace(go.Scatter(
            x=[render_base["bus_pos"]], y=[-0.15], mode='markers+text',
            name='Heuristic (Base)',
            text=["🚌"], textposition="bottom center",
            marker=dict(size=22, color='#475569', line=dict(width=2, color='#94a3b8')),
        ))
    
    fig.add_trace(go.Scatter(
        x=[render_rl["bus_pos"]], y=[0.15], mode='markers+text',
        name='AI: Strategic Strategy',
        text=["🚀"], textposition="top center",
        marker=dict(size=30, color='#0ea5e9', line=dict(width=3, color='#8b5cf6')),
    ))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=10), height=280,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.7, len(stops)-0.3]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 15]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1, font=dict(size=10, color="#94a3b8")),
        hovermode='closest'
    )
    return fig

def create_telemetry_plot():
    """Modern area charts for reward history."""
    fig = go.Figure()
    if state.reward_history_rl:
        steps = list(range(len(state.reward_history_rl)))
        fig.add_trace(go.Scatter(
            x=steps, y=state.reward_history_rl, name='AI: Strategic Strategy',
            line=dict(color='#10b981', width=4, shape='spline'),
            fill='tozeroy', fillcolor='rgba(16,185,129,0.05)'
        ))
    if state.reward_history_base:
        steps = list(range(len(state.reward_history_base)))
        fig.add_trace(go.Scatter(
            x=steps, y=state.reward_history_base, name='Baseline: Simple Greedy',
            line=dict(color='rgba(148,163,184,0.5)', width=2, dash='dot')
        ))
        
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=10, b=40), height=300,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1, font=dict(size=10)),
        font=dict(family='Inter', color='#64748b', size=10),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
    )
    return fig

# ---------------------------------------------------------------------------
# Global Theme CSS (Apple-Style Premium Dark Mode)
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;500;700;900&display=swap');

:root {
    --apple-bg: #0b0f19;
    --apple-card: rgba(30, 41, 59, 0.7);
    --apple-blue: #0ea5e9;
    --apple-green: #10b981;
    --apple-purple: #8b5cf6;
    --apple-border: rgba(255, 255, 255, 0.08);
}

body { background: var(--apple-bg) !important; color: #f1f5f9 !important; font-family: 'Inter', system-ui, sans-serif; }

.header-box { 
    background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(15,23,42,1)); 
    padding: 35px 30px; border-radius: 24px; border: 1px solid var(--apple-border); 
    display: flex; align-items: center; gap: 25px; box-shadow: 0 20px 50px rgba(0,0,0,0.6); 
    margin-bottom: 25px; position: relative; overflow: hidden;
}
.header-box::after { content: ''; position: absolute; top:0; left:0; right:0; height:1px; background: linear-gradient(90deg, transparent, rgba(14,165,233,0.3), transparent); }

.header-title { margin:0; font-family: 'Outfit', sans-serif; font-weight: 900; letter-spacing: -1px; font-size: 2.8rem; background: linear-gradient(to right, #0ea5e9, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 10px rgba(14,165,233,0.3)); }

.info-box { background: rgba(16,185,129,0.06); padding: 18px; border-radius: 16px; border: 1px solid rgba(16,185,129,0.2); border-left: 5px solid #10b981; }

.perf-card { background: var(--apple-card); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-radius: 20px; padding: 22px; border: 1px solid var(--apple-border); box-shadow: 0 10px 30px rgba(0,0,0,0.2); transition: all 0.3s ease; }
.perf-card:hover { transform: translateY(-5px); border-color: rgba(14,165,233,0.2); box-shadow: 0 15px 40px rgba(0,0,0,0.4); }

.badge { display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-blue { background: rgba(14,165,233,0.15); color: #0ea5e9; border: 1px solid rgba(14,165,233,0.3); }

.metric-val { font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 900; line-height: 1; margin: 8px 0; color: #f8fafc; }
.metric-label { font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }

.xai-box { background: rgba(15, 23, 42, 0.95); border-radius: 20px; border: 1px solid var(--apple-border); box-shadow: 0 10px 40px rgba(0,0,0,0.5); padding: 24px; position:relative; overflow:hidden;}
.xai-title { font-family: 'Outfit', sans-serif; font-size: 1.1rem; color: #cbd5e1; font-weight: 800; letter-spacing: 1px; margin-bottom: 20px; display:flex; align-items:center; gap:10px; }
.xai-title::before { content:''; display:inline-block; width:10px; height:10px; background:#8b5cf6; border-radius:50%; box-shadow: 0 0 10px #8b5cf6; }

.reason-bubble { background: rgba(0, 0, 0, 0.2); padding: 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.03); font-size: 0.9rem; line-height: 1.6; color: #94a3b8; }

#start-btn { height: 60px !important; border-radius: 30px !important; font-size: 1.1rem !important; transition: all 0.3s ease !important; background: linear-gradient(90deg, #0ea5e9, #8b5cf6) !important; color:white !important; border:none !important; font-weight: 800 !important; cursor: pointer !important; }
#start-btn:hover { transform: scale(1.02); box-shadow: 0 0 30px rgba(139,92,246,0.5); }

/* Force clean tables outside of dataframes */
.xai-table { border-collapse: collapse; width: 100%; border:none; }
.xai-table th { color: #64748b; font-size: 0.65rem; text-transform: uppercase; padding: 4px 10px; font-weight: 800; letter-spacing: 1px; border-bottom: 1px solid rgba(255,255,255,0.05); }
.xai-table td { padding: 12px 10px; border-bottom: 1px solid rgba(255,255,255,0.02); }
"""

def get_xai_panel(render_rl: Dict[str, Any]):
    q = state.last_q_values
    best_idx = np.argmax(q)
    
    # Simple Softmax for "Confidence"
    exp_q = np.exp(q - np.max(q))
    probs = exp_q / exp_q.sum()
    confidence = probs[best_idx]

    rows = ""
    for i, act_name in ACTION_MAP.items():
        check = "✓" if i == best_idx else ""
        color = "#22d3ee" if i == best_idx else "rgba(255,255,255,0.2)"
        glow = "text-shadow: 0 0 10px rgba(34,211,238,0.3);" if i == best_idx else ""
        rows += f"""
        <tr style="color: {color}; {glow}">
            <td>{act_name}</td>
            <td style="text-align: right; font-family: 'Outfit'; font-weight:700;">{q[i]:.2f}</td>
            <td style="text-align: right; font-weight: 900; color:#22d3ee; padding-right:15px;">{check}</td>
        </tr>
        """

    return f"""
    <div class="xai-box">
        <b class="xai-title">MULTI-AGENT AI CONTEXT PANEL</b>
        <table class="xai-table">
            <thead>
                <tr>
                    <th>POLICIES</th>
                    <th style="text-align: right;">Q-VALUE</th>
                    <th style="text-align: right; padding-right:15px;">STATUS</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        
        <div class="reason-bubble" style="margin-top:20px;">
            <b style="color: #8b5cf6; display:block; margin-bottom: 8px; font-size: 0.65rem; text-transform:uppercase; letter-spacing:1px;">📜 AI Debate Insight:</b>
            {state.last_reason}
        </div>
    </div>
    """

def get_performance_card():
    """Calculates and returns a high-impact score card with Apple-style badges."""
    if not (state.reward_history_rl and state.reward_history_base and len(state.reward_history_rl) > 1):
        return "<div class='perf-card' style='text-align:center;'>Initializing analytics...</div>"
    
    # Calculate Improvements
    rl_score = state.reward_history_rl[-1]
    bs_score = state.reward_history_base[-1]
    bs_val = abs(bs_score) if bs_score != 0 else 1.0
    improvement_reward = ((rl_score - bs_score) / bs_val) * 100
    
    rl_picked = state.env_rl.total_picked
    bs_picked = state.env_base.total_picked if state.env_base else 1
    improvement_speed = ((rl_picked - bs_picked) / (bs_picked or 1)) * 100
    
    rl_fuel = state.env_rl.total_fuel_used
    bs_fuel = state.env_base.total_fuel_used if state.env_base else 1
    eff_rl = rl_picked / (rl_fuel or 1)
    eff_bs = bs_picked / (bs_fuel or 1)
    improvement_fuel = ((eff_rl - eff_bs) / (eff_bs or 1)) * 100

    def get_card(label, val_raw, imp_val, color_class):
        arrow = "+" if imp_val > 0 else "-"
        # Clean labels
        if label == "REWARD": display_val = f"{val_raw:.0f}"
        elif label == "SPEED": display_val = f"{int(val_raw)} pax"
        else: display_val = f"{val_raw:.2f}"
        
        return f"""
        <div class="perf-card">
            <div class="metric-label">{label}</div>
            <div class="metric-val">{display_val}</div>
            <div class="badge {color_class}">
                {arrow} {abs(imp_val):.0f}% IMPROVEMENT
            </div>
        </div>
        """

    return f"""
    <div style="display: grid; grid-template-columns: 1fr; gap: 15px;">
        {get_card("TASK REWARD", rl_score, improvement_reward, "badge-green")}
        {get_card("SERVICE SPEED", rl_picked, improvement_speed, "badge-blue")}
        {get_card("FUEL EFFICIENCY", eff_rl, improvement_fuel, "badge-green")}
    </div>
    """

# ---------------------------------------------------------------------------
# Logic Engine
# ---------------------------------------------------------------------------

def generate_dynamic_debate(act, obs):
    """Simulates a Multi-Agent AI oversight committee debating the RL action."""
    pos, fuel, onboard, q0, q1, q2, step = obs
    
    traffic_cop = ""
    cust_advocate = ""
    fuel_analyst = ""
    
    if fuel < 20: 
        fuel_analyst = "🚨 CRITICAL: Fuel is severely low. Immediate conservation required."
    else: 
        fuel_analyst = f"✅ Optimal: Fuel at {fuel:.1f}%. Proceed with standard routing."
        
    if q0 > 5: 
        cust_advocate = f"⚠️ High Wait: Stop {int(pos)} has {int(q0)} angry passengers."
    elif q1 > 5: 
        cust_advocate = f"⚠️ High Wait downstream: Next stop is crowded."
    else: 
        cust_advocate = "✅ Wait times are within SLA limits. Service running smoothly."

    if act == 2:
        reason = "RL consensus aligned: Resolving localized bottleneck node."
        if q0 > 8: traffic_cop = "Approving WAIT to clear primary congestion node."
        else: traffic_cop = "Strategic IDLE to aggregate demand and improve downstream flow."
    elif act == 0:
        reason = "RL consensus aligned: Aggressive pickup & progression."
        traffic_cop = "Approving MOVE+PICKUP to preserve network velocity."
    else:
        reason = "RL consensus aligned: Bypassing to optimize global throughput."
        traffic_cop = "Approving SKIP to reach higher density clusters faster."

    return f"""
    <div style="font-size: 0.85rem; line-height: 1.5;">
        <div style="margin-bottom: 6px;"><b style="color:#60a5fa">👮 Network Dispatcher:</b> {traffic_cop}</div>
        <div style="margin-bottom: 6px;"><b style="color:#f87171">🧑‍💼 Customer Success:</b> {cust_advocate}</div>
        <div style="margin-bottom: 8px;"><b style="color:#34d399">🔋 Energy Analyst:</b> {fuel_analyst}</div>
        <hr style="border: 0; height: 1px; background: rgba(255,255,255,0.1); margin: 8px 0;" />
        <div style="color: #fbbf24; font-weight: 800;">🤖 RL Final Decision: {reason}</div>
    </div>
    """

def apply_what_if(stop_idx, add_passengers, sabotage_fuel=False):
    """Modifies the live environment state."""
    n = int(add_passengers)
    idx = int(stop_idx)
    if state.env_rl:
        # Each queue entry is a wait-time int; new passengers start at 0
        state.env_rl.stop_queues[idx].extend([0] * n)
        if sabotage_fuel:
            state.env_rl.fuel = max(0.0, state.env_rl.fuel - 30.0)
            
    if state.env_base:
        state.env_base.stop_queues[idx].extend([0] * n)
        if sabotage_fuel:
            state.env_base.fuel = max(0.0, state.env_base.fuel - 30.0)
            
    return f"Applied: +{add_passengers} pax at S{stop_idx}" + (" | FUEL REDUCED!" if sabotage_fuel else "")

def init_env(difficulty: str, compare: bool, agent_mode: str = "Dueling DDQN (Local)"):
    state.difficulty = difficulty
    state.compare_mode = compare
    state.agent_mode = agent_mode
    
    # Map conceptual names to task IDs
    val = difficulty.lower().strip()
    if val == "easy": task_key = "task1_1"
    elif val == "medium": task_key = "task2_1"
    elif val == "hard": task_key = "task3_1"
    else: task_key = val.replace("-", "_")
    
    task = get_task(task_key)
    
    # Initialize RL Env
    state.env_rl = task.build_env()
    state.obs_rl_model = state.env_rl.reset()
    state.obs_rl = state.obs_rl_model.to_array()
    
    # Initialize Baseline
    if compare:
        state.env_base = task.build_env()
        state.obs_base_model = state.env_base.reset()
        state.obs_base = state.obs_base_model.to_array()
    else:
        state.env_base = None
    
    state.done = False
    state.reward_history_rl = [0.0]
    state.reward_history_base = [0.0] if compare else []
    
    # Initialize agents
    if agent_mode == "LLM Optimizer (OpenRouter)":
        state.agent = LLMAgent()
    else:
        state.agent = HeuristicAgent() # Default fallback
        # Load local DQN if available
        model_paths = [
            DEFAULT_MODEL,
            os.path.join(MODELS_DIR, "dqn_bus_v6_best.pt"),
            "dqn_bus_v6_best.pt",
            os.path.join(MODELS_DIR, "dqn_bus_v5.pt"),
            "dqn_bus_v5.pt"
        ]
        for path in model_paths:
            if os.path.exists(path):
                try:
                    state.agent = DQNAgent.load(path)
                    print(f"Successfully loaded model from: {path}")
                    break
                except Exception: continue
    
    try:
        render_rl = state.env_rl.render()
        render_base = state.env_base.render() if compare else None
        return create_comparison_plot(render_rl, render_base), create_telemetry_plot(), get_xai_panel(render_rl), get_performance_card()
    except Exception as e:
        return create_error_fig(str(e)), create_error_fig("Telemetry Error"), f"<div style='color:red'>Render Error: {e}</div>", ""

def step_env():
    if not state.env_rl or state.done:
        # Auto-init if called while empty
        init_env(state.difficulty, state.compare_mode)
    
    if state.done:
        return (
            create_comparison_plot(state.env_rl.render(), state.env_base.render() if state.compare_mode else None),
            create_telemetry_plot(),
            get_xai_panel(state.env_rl.render()),
            get_performance_card()
        )

    # 1. RL / LLM Agent Decision
    if isinstance(state.agent, LLMAgent):
        q_vals, llm_reason = state.agent.predict_q_values(state.obs_rl)
        state.last_q_values = q_vals
        state.last_reason = llm_reason
    else:
        q_vals = state.agent.predict_q_values(state.obs_rl)
        state.last_q_values = q_vals
        act_rl_raw = int(np.argmax(q_vals))
        state.last_reason = generate_dynamic_debate(act_rl_raw, state.obs_rl)

    act_rl = int(np.argmax(q_vals))
    obs_m_rl, rew_rl, done_rl, _ = state.env_rl.step(act_rl)
    state.obs_rl = obs_m_rl.to_array()
    state.reward_history_rl.append(float(state.env_rl.total_reward))
    
    # 2. Baseline Decision (Simple Greedy)
    render_base = None
    if state.compare_mode and state.env_base:
        # Simple Greedy Heuristic: Wait if q > 5, else Move
        q0_base = len(state.env_base.stop_queues[state.env_base.bus_pos])
        act_base = 2 if q0_base > 5 else 0
        obs_m_base, _, done_base, _ = state.env_base.step(act_base)
        state.obs_base = obs_m_base.to_array()
        state.reward_history_base.append(float(state.env_base.total_reward))
        render_base = state.env_base.render()
        if done_base: state.done = True

    if done_rl: state.done = True
    
    render_rl = state.env_rl.render()
    return (
        create_comparison_plot(render_rl, render_base),
        create_telemetry_plot(),
        get_xai_panel(render_rl),
        get_performance_card()
    )

# ---------------------------------------------------------------------------
# UI Definition
# ---------------------------------------------------------------------------

with gr.Blocks(title="OpenEnv Bus RL Optimizer", theme=gr.themes.Default(primary_hue="cyan")) as demo:
    with gr.Column(elem_classes="header-box"):
        with gr.Row():
            gr.Markdown("# 🚀 TransitFlow AI", elem_classes="header-title")
            with gr.Column():
                gr.Markdown(
                    "**Autonomous Bus Routing Engine** | OpenEnv Compliant [ROUND 1]  \n"
                    "Calibrated with GTFS Transit Data (Mumbai/Pune) for Real-World RL Validation.",
                    elem_classes="info-box"
                )

    with gr.Row(equal_height=False):
        # SIDEBAR: COMMAND CENTER
        with gr.Column(scale=1):
            gr.Markdown("### 📡 SYSTEM TELEMETRY", elem_classes="metric-label")
            perf_card = gr.HTML(get_performance_card())
            
            with gr.Group(elem_classes="perf-card"):
                gr.Markdown("### 🕹️ CONTROL DECK", elem_classes="metric-label")
                agent_sel = gr.Dropdown(
                    choices=["Dueling DDQN (Local)", "LLM Optimizer (OpenRouter)"],
                    value="Dueling DDQN (Local)",
                    label="Agent Brain"
                )
                with gr.Row():
                    test_btn = gr.Button("TEST API CONNECTION", size="sm", variant="secondary")
                    test_status = gr.HTML("<span style='opacity:0.5; font-size:0.7rem;'>Ping OpenRouter to verify key...</span>")
                
                diff = gr.Radio(["easy", "medium", "hard"], label="Complexity", value="medium")
                comp = gr.Checkbox(label="Baseline Benchmarking", value=True)
                start_btn = gr.Button("INITIALIZE NEW SESSION", variant="secondary")
                
                demo_run_btn = gr.Button("DEPLOY AI (AUTORUN)", variant="primary", elem_id="start-btn")

        # MAIN FEED: REAL-TIME OPTIMIZATION
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🛰️ LIVE MONITOR"):
                    plot_area = gr.Plot(create_comparison_plot({"stops": [{"stop_idx": i, "queue_len": 0} for i in range(12)], "bus_pos": 0}), label="Real-Time Network Visualization")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            xai_panel = gr.HTML(get_xai_panel({"q_values": [0]*3, "best_idx": 0}))
                        with gr.Column(scale=1):
                            with gr.Row():
                                step_btn = gr.Button("SINGLE STEP", scale=1)
                                inner_run_btn = gr.Button("RUN 10", variant="secondary", scale=1)
                            
                            with gr.Group(elem_classes="perf-card"):
                                gr.Markdown("### ⚠️ INCIDENT DRILL", elem_classes="metric-label")
                                stop_target = gr.Slider(0, 11, step=1, label="Target Station")
                                pax_add = gr.Slider(0, 20, step=1, label="Inject Demand")
                                sabotage = gr.Checkbox(label="Saboteur: Fuel Leak")
                                apply_btn = gr.Button("INJECT EVENT", variant="secondary")

                with gr.TabItem("📈 PERFORMANCE DATA"):
                    telemetry = gr.Plot(create_telemetry_plot(), label="Optimization Convergence Trends")
                    convergence_plot = gr.Plot(create_convergence_plots(), label="Training Analytics")

    # Log Message
    log_msg = gr.Markdown("*System Status: Initialized Core Engines.*")

    # Wiring
    outputs = [plot_area, telemetry, xai_panel, perf_card]
    
    test_btn.click(test_api_key, None, [test_status])
    start_btn.click(init_env, [diff, comp, agent_sel], outputs)
    apply_btn.click(apply_what_if, [stop_target, pax_add, sabotage], [log_msg])
    step_btn.click(step_env, None, outputs)
    
    def run_sequence(steps, diff_val, comp_val, agent_val):
        if not state.env_rl:
            p, t, x, s = init_env(diff_val, comp_val, agent_val)
            yield p, t, x, s
            time.sleep(0.5)

        for _ in range(steps):
            if state.done: break
            p, t, x, s = step_env()
            yield p, t, x, s
            time.sleep(0.15)
    
    def run_10(d, c, a):
        for res in run_sequence(10, d, c, a): yield res
    
    def run_20(d, c, a):
        for res in run_sequence(20, d, c, a): yield res
    
    inner_run_btn.click(run_10, [diff, comp, agent_sel], outputs)
    demo_run_btn.click(run_20, [diff, comp, agent_sel], outputs)

def main():
    import gradio as gr
    app = gr.mount_gradio_app(api_app, demo, path="/")
    print("Starting OpenEnv Server + Dashboard on http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
