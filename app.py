import streamlit as st
from exa_py import Exa
import joblib
from sentence_transformers import SentenceTransformer, util
import os
import speech_recognition as sr 
import json
import streamlit.components.v1 as components
import re
import sqlite3
from datetime import datetime

# IMPORT YOUR NEW LOCAL BRAIN
from rag_engine import NeuralRAG

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Curio", page_icon="✨", layout="centered", initial_sidebar_state="collapsed")

# --- DATABASE SETUP (Feature 4: Persistent Memory) ---
def init_db():
    conn = sqlite3.connect('search_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            summary TEXT,
            top_link TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def save_to_history(query, summary, top_link):
    # Prevent duplicate saves if query is identical to the very last entry
    conn = sqlite3.connect('search_history.db')
    c = conn.cursor()
    c.execute('SELECT query FROM history ORDER BY id DESC LIMIT 1')
    last_entry = c.fetchone()
    
    if not last_entry or last_entry[0] != query:
        c.execute('INSERT INTO history (query, summary, top_link, timestamp) VALUES (?, ?, ?, ?)', 
                  (query, summary, top_link, datetime.now()))
        conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('search_history.db')
    c = conn.cursor()
    c.execute('SELECT query, summary, top_link, timestamp FROM history ORDER BY id DESC LIMIT 10')
    data = c.fetchall()
    conn.close()
    return data

# Initialize DB on load
init_db()

# --- HTML TEMPLATE FOR GRAPH (Feature 2: Visuals) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #ffffff; margin: 0; overflow: hidden; }
        #graph-container { transition: all 0.3s ease-in-out; }
        .tooltip {
            position: absolute; text-align: center; padding: 8px; font-size: 12px;
            background: rgba(0, 0, 0, 0.8); color: white; border-radius: 4px;
            pointer-events: none; opacity: 0; transition: opacity 0.2s; z-index: 100;
        }
        circle { stroke: #fff; stroke-width: 2px; cursor: pointer; transition: all 0.2s; }
        circle:hover { stroke: #333; stroke-width: 3px; }
        text { font-family: 'Inter', sans-serif; pointer-events: none; text-shadow: 0 1px 0 #fff; }
    </style>
</head>
<body class="flex items-center justify-center h-screen w-screen p-0">
    <div id="graph-card" class="bg-white flex flex-col w-full h-full relative overflow-hidden">
        
        <!-- Header inside the Iframe -->
        <div class="flex items-center justify-between px-4 py-2 border-b border-gray-100 bg-gray-50 z-10">
            <div>
                <h2 class="text-sm font-bold text-gray-800"><i class="fas fa-network-wired text-indigo-500 mr-2"></i>Results Cluster</h2>
            </div>
            <div class="flex gap-2">
                <button onclick="resetZoom()" class="p-1 text-gray-500 hover:text-indigo-600 rounded" title="Reset View">
                    <i class="fas fa-compress-arrows-alt"></i>
                </button>
                <button onclick="toggleFullScreen()" class="p-1 text-gray-500 hover:text-indigo-600 rounded" title="Maximize/Minimize">
                    <i id="max-btn-icon" class="fas fa-expand"></i>
                </button>
            </div>
        </div>

        <div id="graph-container" class="flex-grow bg-slate-50 relative w-full h-full cursor-move">
            <div id="tooltip" class="tooltip"></div>
        </div>

        <div class="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm p-2 rounded-lg shadow-lg border border-gray-100 text-xs z-10 pointer-events-none">
            <div class="font-semibold mb-1 text-gray-600">Categories</div>
            <div class="flex flex-col gap-1" id="legend-content"></div>
        </div>
    </div>

    <script>
        // INJECTED DATA FROM PYTHON
        const data = {{DATA_JSON}};

        // Define specific colors for your classifier categories
        const colors = {
            "Development": "#6366f1", "Design": "#ec4899", "News": "#f59e0b", 
            "Social": "#10b981", "Tools": "#3b82f6", "Research": "#8b5cf6",
            "Business": "#14b8a6", "Marketing": "#f43f5e"
        };
        const defaultPalette = ["#6366f1", "#ec4899", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6"];

        function getColor(group) {
            if (colors[group]) return colors[group];
            let hash = 0;
            for (let i = 0; i < group.length; i++) hash = group.charCodeAt(i) + ((hash << 5) - hash);
            return defaultPalette[Math.abs(hash) % defaultPalette.length];
        }

        let width, height, svg, g, simulation, zoom;
        let container = document.getElementById('graph-container');
        let card = document.getElementById('graph-card');
        let tooltip = document.getElementById('tooltip');

        function initGraph() {
            d3.select("#graph-container svg").remove();
            width = container.clientWidth;
            height = container.clientHeight;

            zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", (e) => g.attr("transform", e.transform));
            svg = d3.select("#graph-container").append("svg")
                .attr("width", width).attr("height", height)
                .call(zoom).on("dblclick.zoom", null);
            g = svg.append("g");

            simulation = d3.forceSimulation(data.nodes)
                .force("charge", d3.forceManyBody().strength(5))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(d => d.val + 5).iterations(2))
                .force("x", d3.forceX(width / 2).strength(0.08))
                .force("y", d3.forceY(height / 2).strength(0.08));

            const node = g.append("g").selectAll("g").data(data.nodes).join("g")
                .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));

            node.append("circle")
                .attr("r", d => d.val)
                .attr("fill", d => getColor(d.group))
                .attr("opacity", 0.9)
                .on("click", (e, d) => { if(d.url) window.open(d.url, '_blank'); })
                .on("mouseover", function(e, d) {
                    d3.select(this).transition().duration(200).attr("r", d.val + 5);
                    showTooltip(e, d);
                })
                .on("mouseout", function(e, d) {
                    d3.select(this).transition().duration(200).attr("r", d.val);
                    hideTooltip();
                });

            node.append("text").text(d => d.label)
                .attr("y", d => d.val + 12).attr("text-anchor", "middle")
                .attr("font-size", "10px").attr("font-weight", "600").attr("fill", "#374151")
                .style("pointer-events", "none");

            simulation.on("tick", () => node.attr("transform", d => `translate(${d.x},${d.y})`));

            // Legend
            const legend = document.getElementById('legend-content');
            legend.innerHTML = '';
            const groups = [...new Set(data.nodes.map(d => d.group))];
            groups.forEach(key => {
                legend.innerHTML += `<div class="flex items-center gap-2"><div class="w-3 h-3 rounded-full" style="background-color: ${getColor(key)}"></div><span>${key}</span></div>`;
            });
        }

        // --- INTERACTION LOGIC ---
        function dragstarted(e, d) { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
        function dragged(e, d) { d.fx = e.x; d.fy = e.y; }
        function dragended(e, d) { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }
        
        function showTooltip(e, d) { 
            tooltip.style.opacity = 1; tooltip.innerHTML = `<strong>${d.full_title}</strong><br><span style="color:#ccc">${d.domain}</span>`;
            tooltip.style.left = (e.pageX + 10) + 'px'; tooltip.style.top = (e.pageY - 28) + 'px';
        }
        function hideTooltip() { tooltip.style.opacity = 0; }
        
        function resetZoom() { svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity); }

        function toggleFullScreen() {
            if (!document.fullscreenElement) {
                card.requestFullscreen().catch(err => {
                    alert(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
                });
            } else {
                document.exitFullscreen();
            }
        }
        
        document.addEventListener('fullscreenchange', (event) => {
            const icon = document.getElementById('max-btn-icon');
            if (document.fullscreenElement) {
                icon.classList.remove('fa-expand');
                icon.classList.add('fa-compress');
            } else {
                icon.classList.remove('fa-compress');
                icon.classList.add('fa-expand');
            }
            setTimeout(() => {
                width = container.clientWidth;
                height = container.clientHeight;
                svg.attr("width", width).attr("height", height);
                simulation.force("center", d3.forceCenter(width / 2, height / 2));
                simulation.alpha(0.3).restart();
            }, 100);
        });

        window.addEventListener('load', initGraph);
        new ResizeObserver(() => { if(svg) { width = container.clientWidth; height = container.clientHeight; svg.attr("width", width).attr("height", height); simulation.force("center", d3.forceCenter(width / 2, height / 2)); simulation.alpha(0.3).restart(); } }).observe(container);
    </script>
</body>
</html>
"""

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1 { font-family: 'Comic Sans MS', sans-serif; color: #FF4B4B; }
    .stTextInput > div > div > input { background-color: #262730; color: #FAFAFA; border: 2px solid #FF4B4B; border-radius: 20px; }
    .result-card-container { background-color: #262730; padding: 15px; border-radius: 15px; border-left: 5px solid #FF4B4B; }
    .ml-tag { background-color: #7000FF; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.75em; font-weight: bold; }
    .domain-tag { background-color: #3E3E3E; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; color: #E0E0E0; margin-right: 10px;}
    .citation-tag { color: #00e676; font-weight: bold; font-size: 0.8em; cursor: help; vertical-align: super; }
    .warning-tag { color: #ff9100; font-weight: bold; font-size: 0.8em; vertical-align: super; }
    a { color: #FF4B4B !important; text-decoration: none; font-weight: bold; }
    img { border-radius: 10px; object-fit: cover; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODELS (Cached) ---
@st.cache_resource
def load_models():
    # 1. Load Neural Classifier
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        classifier = joblib.load('neural_classifier.pkl')
    except:
        embedder, classifier = None, None
    
    # 2. Load RAG Engine
    rag = NeuralRAG()
    
    return embedder, classifier, rag

embedder, classifier, rag = load_models()

# --- 3. HELPER FUNCTION: Safety Check (Feature 3) ---
def perform_safety_check(summary, results, embedder):
    """
    Checks AI summary against sources. Returns annotated text and a safety score (0-100).
    """
    if not embedder or not summary:
        return summary, 0.0

    # Split summary into sentences
    summary_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', summary) if s.strip()]
    
    # Prepare Source Embeddings
    source_chunks = []
    source_map = [] 
    
    for idx, result in enumerate(results):
        if not result.text: continue
        # Chunking: Break large text into sentences/chunks for better matching
        chunks = [c.strip() for c in re.split(r'(?<=[.!?]) +', result.text[:2000]) if len(c) > 20]
        for chunk in chunks:
            source_chunks.append(chunk)
            source_map.append(idx + 1) # Source ID [1], [2]...

    if not source_chunks:
        return summary, 0.0

    # Vectorize
    source_embeddings = embedder.encode(source_chunks, convert_to_tensor=True)
    
    annotated_summary = ""
    total_score = 0
    verified_count = 0
    
    for sentence in summary_sentences:
        sent_embedding = embedder.encode(sentence, convert_to_tensor=True)
        
        # Check similarity against all source chunks
        cos_scores = util.cos_sim(sent_embedding, source_embeddings)[0]
        best_score_idx = int(cos_scores.argmax())
        best_score = float(cos_scores[best_score_idx])
        
        source_id = source_map[best_score_idx]
        
        # Threshold: 0.45 is a balanced threshold for MiniLM
        if best_score > 0.45:
            annotated_summary += f"{sentence} <span class='citation-tag'>[{source_id}]</span> "
            total_score += best_score
            verified_count += 1
        else:
            annotated_summary += f"{sentence} <span class='warning-tag'>[Unverified]</span> "
            
    final_score = (verified_count / len(summary_sentences)) * 100 if summary_sentences else 0
    return annotated_summary, final_score

# --- 4. EXA SETUP ---
api_key = '5e8e9d1d-2b0c-41b1-b00d-4b3aed1ef247'
exa = Exa(api_key)

# --- 5. SESSION STATE ---
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""
if 'last_voice_query' not in st.session_state:
    st.session_state.last_voice_query = ""

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # --- Feature 4: History Sidebar ---
    st.subheader("📜 Search History")
    history_items = get_history()
    
    if history_items:
        for item in history_items:
            q_text, _, _, ts = item
            # Parse timestamp to show nicely (e.g., 14:30)
            try:
                time_label = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M")
            except:
                time_label = "Recent"
                
            # Button to reload query
            if st.button(f"🕒 {time_label}: {q_text[:15]}...", key=f"hist_{ts}"):
                st.session_state.query_text = q_text
                st.rerun()
    else:
        st.caption("No history yet.")
    
    st.markdown("---")
    
    # VOICE INPUT
    st.write("🎤 **Voice Search**")
    audio_value = st.audio_input("Record Voice Query")

    if audio_value:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(audio_value) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                
                if text and text != st.session_state.last_voice_query:
                    st.session_state.last_voice_query = text
                    st.session_state.query_text = text
                    st.success(f"Heard: {text}")
                    st.rerun()
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    
    target_lang = st.selectbox(
        "Output Language:",
        options=['en', 'hi', 'es', 'fr', 'de', 'ja'],
        format_func=lambda x: {'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese'}[x]
    )
    
    filter_date = st.date_input("Published After:", value=None)
    
    st.divider()
    if classifier: st.success("✅ Neural Classifier Active")
    if rag: st.success("✅ Local RAG Brain Active")

# --- 7. MAIN UI ---
st.title("✨ Curio")
st.markdown("### Powered by Exa & **Local Generative AI**")

query = st.text_input("What are you curious about?", value=st.session_state.query_text, placeholder="Type or use Voice Search in sidebar...")

if query:
    if query != st.session_state.query_text:
        st.session_state.query_text = query
        
    start_date = filter_date.strftime("%Y-%m-%d") if filter_date else None
    
    # --- STEP 1: RETRIEVAL ---
    with st.spinner("🔍 Searching & Analyzing..."):
        try:
            response = exa.search_and_contents(
                query, type="neural", num_results=5, text=True, start_published_date=start_date
            )
        except Exception as e:
            st.error(f"Search Error: {e}")
            response = None

    if response:
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📝 AI Explanation", "🕸️ Interactive Map", "🛡️ Safety & Citations"])
        
        # --- TAB 1: GENERATION ---
        with tab1:
            st.subheader(f"AI Explanation ({target_lang})")
            
            with st.spinner("🧠 Generating Explanation..."):
                try:
                    summary = rag.generate_rag_summary(query, response.results, target_lang=target_lang)
                    
                    # --- AUTO-SAVE (Feature 4) ---
                    top_link = response.results[0].url if response.results else ""
                    save_to_history(query, summary, top_link)
                    
                    with st.chat_message("assistant"):
                        st.write(summary)
                        
                        audio_file = rag.text_to_speech(summary, filename="response.mp3")
                        if audio_file:
                            st.audio(audio_file)
                        
                except Exception as e:
                    st.warning(f"Summarization failed: {e}")

            st.markdown("---")
            st.subheader("🌍 Verified Sources")
            
            for idx, result in enumerate(response.results):
                source_id = idx + 1
                domain = result.url.split('/')[2].replace('www.', '')
                image_url = getattr(result, "image", None)
                
                # CLASSIFICATION
                ml_tag_html = ""
                if embedder and classifier:
                    try:
                        context_text = f"{result.title} {result.text[:100] if result.text else ''}"
                        vector = embedder.encode([context_text])
                        prediction = classifier.predict(vector)[0]
                        ml_tag_html = f"<span class='ml-tag'>{prediction}</span>"
                    except:
                        pass

                # TRANSLATION
                display_text = result.text[:400] + "..." if result.text else "No preview"
                if target_lang != 'en' and result.text:
                    display_text = rag.translate_text(result.text[:400], target_lang) + "..."

                with st.container():
                    st.markdown(f"#### [{source_id}] {result.title}")
                    st.markdown(f"{ml_tag_html} <span class='domain-tag'>🏛 {domain}</span>", unsafe_allow_html=True)
                    
                    if image_url and (image_url.startswith("http://") or image_url.startswith("https://")):
                         st.image(image_url, width=200)

                    st.markdown(f"[🔗 Visit Website]({result.url})")
                    with st.expander(f"Read Preview ({target_lang})"):
                        st.write(display_text)
                    st.markdown("---")

        # --- TAB 2: INTERACTIVE BUBBLE GRAPH ---
        with tab2:
            st.subheader("🕸️ Web Resource Cluster")
            
            graph_nodes = []
            graph_nodes.append({
                "id": "query", "label": "Query", "full_title": query,
                "group": "Query", "val": 45, "url": "", "domain": "Search Term"
            })
            
            for i, result in enumerate(response.results):
                group = "General"
                if embedder and classifier:
                    try:
                        context_text = f"{result.title} {result.text[:100] if result.text else ''}"
                        vector = embedder.encode([context_text])
                        prediction = classifier.predict(vector)[0]
                        group = prediction
                    except: pass
                
                domain = result.url.split('/')[2].replace('www.', '')
                graph_nodes.append({
                    "id": f"res_{i}", "label": domain, "full_title": result.title,
                    "group": group, "val": 30, "url": result.url, "domain": domain
                })
            
            graph_data = {"nodes": graph_nodes}
            html_code = HTML_TEMPLATE.replace("{{DATA_JSON}}", json.dumps(graph_data))
            components.html(html_code, height=600, scrolling=False)

        # --- TAB 3: AI SAFETY & CITATIONS (Feature 3) ---
        with tab3:
            st.subheader("🛡️ AI Safety & Fact Verification")
            st.info("This module uses Semantic Embeddings to verify if the AI's summary is grounded in the source text.")
            
            if 'summary' in locals() and embedder:
                with st.spinner("Running Safety Checks..."):
                    # This calls the helper function defined at the top
                    annotated_text, safety_score = perform_safety_check(summary, response.results, embedder)
                
                # Display Score
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("Safety Score", f"{int(safety_score)}%")
                with col2:
                    if safety_score > 80:
                        st.success("✅ High Confidence: Summary is well-supported by sources.")
                        st.progress(safety_score / 100)
                    elif safety_score > 50:
                        st.warning("⚠️ Moderate Confidence: Some claims may need manual verification.")
                        st.progress(safety_score / 100)
                    else:
                        st.error("🚨 Low Confidence: High risk of hallucination.")
                        st.progress(safety_score / 100)
                
                st.markdown("### 📝 Verified Summary")
                st.markdown(annotated_text, unsafe_allow_html=True)
                
                st.markdown("---")
                st.caption("Green numbers like **[1]** indicate the source ID that supports the sentence. **[Unverified]** means no close match was found in the retrieval window.")
            else:
                st.warning("Please generate a summary in Tab 1 first, or ensure models are loaded.")