import streamlit as st
from exa_py import Exa
from datetime import datetime

# --- 1. Page & Theme Configuration ---
st.set_page_config(
    page_title="Curio",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Mode & Images
st.markdown("""
    <style>
    /* Main Background - Dark */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Title Font */
    h1 {
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif; 
        color: #FF4B4B; 
    }
    
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 20px;
        border: 2px solid #FF4B4B;
        padding: 10px 20px;
    }
    
    /* Result Card Styling */
    .result-card-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Domain Tag */
    .domain-tag {
        font-size: 0.8em;
        color: #E0E0E0;
        background-color: #3E3E3E;
        padding: 2px 8px;
        border-radius: 10px;
        margin-right: 10px;
    }
    
    /* Links */
    a {
        color: #FF4B4B !important;
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        text-decoration: underline;
    }
    
    /* Image styling within Streamlit */
    img {
        border-radius: 10px;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Initialize Backend (SECURE FOR DEPLOYMENT) ---
try:
    # This securely fetches the key from Streamlit Cloud Secrets (or local .streamlit/secrets.toml)
    api_key = st.secrets["EXA_API_KEY"]
    exa = Exa(api_key)
except Exception as e:
    # This error message is normal if you haven't set up the secrets file locally yet!
    st.error("⚠️ API Key not found! If running locally, check your .streamlit/secrets.toml file. If deploying, check your App Settings > Secrets.")
    st.stop()

# --- 3. Sidebar Options ---
with st.sidebar:
    st.header("⚙️ Search Filters")
    filter_date = st.date_input("Published After:", value=None)
    st.caption("Leave empty to search all time.")

# --- 4. Main UI ---
st.title("✨ Curio ✨")
st.markdown("### Ask anything. I'll summarize it & show you visuals!")

query = st.text_input("What are you curious about?", placeholder="e.g. The most beautiful libraries in the world")

if query:
    # --- STEP 1: The AI Summary ---
    st.subheader("📝 AI Summary")
    with st.spinner("Curating a summary for you..."):
        try:
            summary_response = exa.answer(query)
            final_text = getattr(summary_response, "answer", str(summary_response))
            st.info(final_text)
                
        except Exception as e:
            st.warning("Could not generate a summary. Showing search results below.")
            print(f"Summary Error: {e}")

    st.markdown("---")

    # --- STEP 2: The Search Results (With Images) ---
    st.subheader("🌍 Explore the Web")
    with st.spinner("Fetching best links & images..."):
        try:
            start_date_str = filter_date.strftime("%Y-%m-%d") if filter_date else None

            # We assume Exa returns an 'image' field in the results automatically for many sites.
            response = exa.search_and_contents(
                query,
                type="neural",
                num_results=5,
                text=True,
                start_published_date=start_date_str 
            )

            for result in response.results:
                domain_name = result.url.split('/')[2].replace('www.', '')
                
                # Check if the result has an image
                image_url = getattr(result, "image", None)

                # --- CARD UI START ---
                with st.container():
                    # We create a simplified "Card" using columns
                    # Col 1: Image (if available), Col 2: Content
                    
                    if image_url:
                        col1, col2 = st.columns([1, 3]) # Image takes 1/4th width
                        
                        with col1:
                            st.image(image_url, use_container_width=True)
                            
                        with col2:
                            st.markdown(f"### {result.title}")
                            st.markdown(f"<span class='domain-tag'>🏛 {domain_name}</span>", unsafe_allow_html=True)
                            st.markdown(f"[🔗 Visit Website]({result.url})")
                    else:
                        # Fallback layout if no image
                        st.markdown(f"### {result.title}")
                        st.markdown(f"<span class='domain-tag'>🏛 {domain_name}</span>", unsafe_allow_html=True)
                        st.markdown(f"[🔗 Visit Website]({result.url})")

                    # Preview Dropdown (Outside columns, full width)
                    with st.expander(f"📖 Read Preview"):
                        if result.text:
                            st.write(result.text[:500] + "...")
                        else:
                            st.caption("No text preview available.")
                    
                    st.markdown("---") 
                # --- CARD UI END ---

        except Exception as e:
            st.error(f"Search failed: {e}")