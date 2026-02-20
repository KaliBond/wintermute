"""
CLD Studio: Interactive Streamlit app for generating Causal Loop Diagrams

Features:
- Two-stage generation: Aha! Paradox → Analyst CLD → Audience Skins
- Base 8-node CAMS lattice as canonical foundation
- Multiple audience skins (Policy brief, Town/kids, Boardroom)
- Perplexity API integration
- CLDai visualization engine rendering
- Export to JSON, HTML, PNG
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

# Import custom modules
from src.cld_architect import CLDArchitect, CAMS_LATTICE, AHA_FRAMEWORK
from src.cld_skins import CLDSkinTranslator, SkinParameters, PRESET_SKINS

# Page config
st.set_page_config(
    page_title="CLD Studio",
    layout="wide",
    page_icon="🌀",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stTextInput > div > div > input {background-color: #1e2130; color: #fafafa;}
    .stTextArea > div > div > textarea {background-color: #1e2130; color: #fafafa;}
    .stSelectbox > div > div > div {background-color: #1e2130; color: #fafafa;}
    h1 {color: #3498db;}
    h2 {color: #e74c3c;}
    h3 {color: #f39c12;}
    .success-box {
        background-color: #1e4d2b;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #27ae60;
        margin: 10px 0;
    }
    .info-box {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #4d3d1e;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #f39c12;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_aha' not in st.session_state:
    st.session_state.generated_aha = None
if 'generated_analyst_cld' not in st.session_state:
    st.session_state.generated_analyst_cld = None
if 'generated_skins' not in st.session_state:
    st.session_state.generated_skins = {}
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'analyst'

# Title
st.title("🌀 CLD Studio")
st.markdown("**Generate Causal Loop Diagrams with Aha! Paradox Framework**")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# API Key
api_key = st.sidebar.text_input(
    "Perplexity API Key",
    type="password",
    value=os.getenv("PERPLEXITY_API_KEY", ""),
    help="Get your API key from perplexity.ai"
)

if api_key:
    os.environ["PERPLEXITY_API_KEY"] = api_key
    st.sidebar.success("✓ API key set")
else:
    st.sidebar.warning("⚠ API key required for generation")

st.sidebar.markdown("---")

# Model selection
model = st.sidebar.selectbox(
    "Perplexity Model",
    ["sonar-pro", "sonar", "sonar-reasoning"],
    index=0,
    help="sonar-pro recommended for best quality"
)

# Collision domain
collision_domain = st.sidebar.selectbox(
    "Collision Domain",
    ["all", "metabolic", "mythic", "executive", "productive"],
    index=0,
    help="Which CAMS subsystem to focus on"
)

st.sidebar.markdown("---")

# Show CAMS lattice
with st.sidebar.expander("📊 CAMS Base Lattice (8 nodes)"):
    for node in CAMS_LATTICE:
        st.markdown(f"**{node['label']}** ({node['domain']})")
        st.caption(node['description'])
        st.markdown("---")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Builder", "📊 Analyst View", "🎨 Audience Skins", "💾 Export"])

# ==================== TAB 1: Builder ====================
with tab1:
    st.header("CLD Builder")
    st.markdown("Generate a new Causal Loop Diagram using the Aha! Paradox framework")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. Define Your Paradox")

        topic = st.text_input(
            "Topic",
            placeholder="e.g., Market Efficiency and Financial Instability",
            help="Brief title for your paradox"
        )

        narrative = st.text_area(
            "Narrative Context",
            placeholder="Describe the conventional wisdom and the paradoxical outcome you want to explore...",
            height=120,
            help="Background context for the paradox (2-4 sentences)"
        )

        must_include_text = st.text_area(
            "Must Include (one per line)",
            placeholder="Price discovery\nLiquidity constraints\nCascade failures",
            height=100,
            help="Key concepts that must appear in the analysis"
        )

        must_include = [line.strip() for line in must_include_text.split('\n') if line.strip()]

    with col2:
        st.subheader("Aha! Framework")
        st.markdown("**7 Sections:**")
        for section in AHA_FRAMEWORK:
            st.markdown(f"**{section.section}** ({section.length})")
            st.caption(section.focus)
            st.markdown("")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Generate Aha! Paradox", type="primary", disabled=not (api_key and topic and narrative)):
            with st.spinner("Generating Aha! Paradox..."):
                try:
                    architect = CLDArchitect(api_key)
                    aha = architect.generate_aha_paradox(topic, narrative, must_include, model)
                    st.session_state.generated_aha = aha
                    st.success("✓ Aha! Paradox generated!")
                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")

    with col2:
        if st.button("🎯 Generate Analyst CLD", type="primary",
                    disabled=not (api_key and st.session_state.generated_aha)):
            with st.spinner(f"Generating analyst CLD (domain: {collision_domain})..."):
                try:
                    architect = CLDArchitect(api_key)
                    cld = architect.generate_analyst_cld(
                        st.session_state.generated_aha,
                        collision_domain,
                        model
                    )
                    st.session_state.generated_analyst_cld = cld
                    st.success("✓ Analyst CLD generated!")
                except Exception as e:
                    st.error(f"❌ CLD generation failed: {str(e)}")

    # Display generated Aha! Paradox
    if st.session_state.generated_aha:
        st.markdown("---")
        st.subheader("Generated Aha! Paradox")

        aha = st.session_state.generated_aha
        sections = [
            ("Anchor", "anchor"),
            ("Default", "default"),
            ("Bottleneck", "bottleneck"),
            ("Collision", "collision"),
            ("Reversal", "reversal"),
            ("Commitment Filter", "commitment_filter"),
            ("Kinetic Result", "kinetic_result")
        ]

        for section_name, section_key in sections:
            if section_key in aha:
                st.markdown(f"**{section_name}:**")
                st.info(aha[section_key])

# ==================== TAB 2: Analyst View ====================
with tab2:
    st.header("Analyst CLD (Canonical Model)")

    if st.session_state.generated_analyst_cld:
        cld = st.session_state.generated_analyst_cld

        # Metadata
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodes", len(cld.nodes))
        col2.metric("Edges", len(cld.edges))
        col3.metric("Loops", len(cld.loops))
        col4.metric("Archetypes", len(cld.archetypes))

        # Archetypes
        if cld.archetypes:
            st.subheader("System Archetypes")
            for arch in cld.archetypes:
                st.markdown(f"- {arch}")

        # Loops
        st.subheader("Feedback Loops")
        for loop in cld.loops:
            loop_type = "🔴 Reinforcing" if loop.polarity == "R" else "🟢 Balancing"
            st.markdown(f"**Loop {loop.loop_id}** ({loop_type}): {loop.label}")
            st.caption(f"Path: {' → '.join(loop.nodes)}")
            if loop.archetype:
                st.caption(f"Archetype: {loop.archetype}")

        # Nodes table
        st.subheader("Nodes")
        nodes_data = []
        for node in cld.nodes:
            nodes_data.append({
                "ID": node.id,
                "Label": node.label,
                "Type": node.type,
                "Domain": node.domain or "N/A",
                "Description": node.description[:80] + "..." if len(node.description) > 80 else node.description
            })
        st.dataframe(pd.DataFrame(nodes_data), use_container_width=True)

        # Edges table
        st.subheader("Causal Edges")
        edges_data = []
        for edge in cld.edges:
            edges_data.append({
                "From": edge.from_node,
                "To": edge.to_node,
                "Polarity": edge.polarity,
                "Width": edge.width,
                "Lag": edge.lag,
                "Description": edge.description[:60] + "..." if len(edge.description) > 60 else edge.description
            })
        st.dataframe(pd.DataFrame(edges_data), use_container_width=True)

        # Render visualization
        st.subheader("Interactive Visualization")
        render_cld_visualization(cld)

    else:
        st.info("👈 Generate an Analyst CLD in the Builder tab first")

# ==================== TAB 3: Audience Skins ====================
with tab3:
    st.header("Audience Skins")
    st.markdown("Translate analyst CLD into audience-specific formats")

    if st.session_state.generated_analyst_cld:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Skin Configuration")

            skin_type = st.selectbox(
                "Audience Type",
                ["policy_brief", "town", "boardroom"],
                format_func=lambda x: {
                    "policy_brief": "Policy Brief",
                    "town": "Town Hall / General Public",
                    "boardroom": "Boardroom / Executive"
                }[x]
            )

            if skin_type == "policy_brief":
                jurisdiction = st.selectbox(
                    "Jurisdiction",
                    ["Federal", "State", "Municipal", "International"],
                    index=0
                )
                time_horizon = st.text_input(
                    "Time Horizon",
                    value="2026-2030"
                )
                audience = st.text_input(
                    "Specific Audience",
                    value="Congressional staffers and federal agency leads"
                )
                reading_level = st.slider("Reading Level (grade)", 8, 16, 14)

                params = SkinParameters(
                    audience_type=skin_type,
                    jurisdiction=jurisdiction,
                    time_horizon=time_horizon,
                    audience=audience,
                    reading_level=reading_level
                )

            elif skin_type == "town":
                audience = st.text_input(
                    "Specific Audience",
                    value="General public at community meeting"
                )
                reading_level = st.slider("Reading Level (grade)", 6, 12, 8)

                params = SkinParameters(
                    audience_type=skin_type,
                    audience=audience,
                    reading_level=reading_level
                )

            else:  # boardroom
                time_horizon = st.text_input(
                    "Time Horizon",
                    value="Strategic (3-5 years)"
                )
                audience = st.text_input(
                    "Specific Audience",
                    value="Corporate executives and board members"
                )
                reading_level = st.slider("Reading Level (grade)", 12, 18, 14)

                params = SkinParameters(
                    audience_type=skin_type,
                    time_horizon=time_horizon,
                    audience=audience,
                    reading_level=reading_level
                )

            if st.button("🎨 Generate Skin", type="primary", disabled=not api_key):
                with st.spinner(f"Translating to {skin_type}..."):
                    try:
                        translator = CLDSkinTranslator(api_key)
                        skinned_cld = translator.apply_skin(
                            st.session_state.generated_analyst_cld,
                            params,
                            model
                        )
                        skin_id = f"{skin_type}_{datetime.now().strftime('%H%M%S')}"
                        st.session_state.generated_skins[skin_id] = skinned_cld
                        st.session_state.current_view = skin_id
                        st.success(f"✓ {skin_type.replace('_', ' ').title()} skin generated!")
                    except Exception as e:
                        st.error(f"❌ Skin generation failed: {str(e)}")

            # Preset skins
            st.markdown("---")
            st.subheader("Preset Skins")
            preset_options = list(PRESET_SKINS.keys())
            selected_preset = st.selectbox(
                "Quick Apply",
                [""] + preset_options,
                format_func=lambda x: x.replace('_', ' ').title() if x else "Select preset..."
            )

            if selected_preset and st.button("Apply Preset"):
                with st.spinner(f"Applying {selected_preset}..."):
                    try:
                        translator = CLDSkinTranslator(api_key)
                        skinned_cld = translator.apply_skin(
                            st.session_state.generated_analyst_cld,
                            PRESET_SKINS[selected_preset],
                            model
                        )
                        skin_id = f"{selected_preset}_{datetime.now().strftime('%H%M%S')}"
                        st.session_state.generated_skins[skin_id] = skinned_cld
                        st.session_state.current_view = skin_id
                        st.success(f"✓ {selected_preset.replace('_', ' ').title()} applied!")
                    except Exception as e:
                        st.error(f"❌ Preset application failed: {str(e)}")

        with col2:
            st.subheader("Generated Skins")

            if st.session_state.generated_skins:
                # View selector
                view_options = ["analyst"] + list(st.session_state.generated_skins.keys())
                current_view = st.selectbox(
                    "View",
                    view_options,
                    index=view_options.index(st.session_state.current_view) if st.session_state.current_view in view_options else 0,
                    format_func=lambda x: "Analyst (Canonical)" if x == "analyst" else x.replace('_', ' ').title()
                )

                if current_view == "analyst":
                    cld_to_show = st.session_state.generated_analyst_cld
                else:
                    cld_to_show = st.session_state.generated_skins[current_view]

                # Show comparison
                st.markdown("**Node Label Comparison:**")
                comparison_data = []
                analyst_cld = st.session_state.generated_analyst_cld
                for i, (analyst_node, skin_node) in enumerate(zip(analyst_cld.nodes, cld_to_show.nodes)):
                    if analyst_node.label != skin_node.label:
                        comparison_data.append({
                            "Analyst": analyst_node.label,
                            "Skinned": skin_node.label
                        })

                if comparison_data:
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                else:
                    st.caption("(No label changes)")

                # Render
                st.markdown("---")
                render_cld_visualization(cld_to_show)

            else:
                st.info("Generate a skin using the controls on the left")

    else:
        st.info("👈 Generate an Analyst CLD in the Builder tab first")

# ==================== TAB 4: Export ====================
with tab4:
    st.header("Export & Download")

    if st.session_state.generated_analyst_cld:
        export_options = ["analyst"] + list(st.session_state.generated_skins.keys())
        export_selection = st.selectbox(
            "Select model to export",
            export_options,
            format_func=lambda x: "Analyst (Canonical)" if x == "analyst" else x.replace('_', ' ').title()
        )

        if export_selection == "analyst":
            export_cld = st.session_state.generated_analyst_cld
        else:
            export_cld = st.session_state.generated_skins[export_selection]

        # Convert to JSON
        from dataclasses import asdict
        export_json = {
            "metadata": export_cld.metadata,
            "nodes": [asdict(n) for n in export_cld.nodes],
            "edges": [asdict(e) for e in export_cld.edges],
            "loops": [asdict(l) for l in export_cld.loops],
            "archetypes": export_cld.archetypes,
            "aha_paradox": export_cld.aha_paradox
        }

        # JSON download
        st.subheader("📄 JSON Export")
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(export_json, indent=2),
            file_name=f"cld_{export_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # HTML download (standalone visualization)
        st.subheader("🌐 HTML Export")
        html_content = generate_standalone_html(export_cld)
        st.download_button(
            "⬇️ Download HTML (standalone)",
            data=html_content,
            file_name=f"cld_{export_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )

        # Preview JSON
        with st.expander("📋 Preview JSON"):
            st.json(export_json)

    else:
        st.info("👈 Generate an Analyst CLD in the Builder tab first")

# ==================== Helper Functions ====================

def render_cld_visualization(cld):
    """Render CLD using CLDai visualization engine"""
    from dataclasses import asdict

    cld_json = {
        "metadata": cld.metadata,
        "nodes": [asdict(n) for n in cld.nodes],
        "edges": [asdict(e) for e in cld.edges],
        "loops": [asdict(l) for l in cld.loops],
        "archetypes": cld.archetypes,
        "aha_paradox": cld.aha_paradox
    }

    # Read template
    template_path = Path(__file__).parent / "templates" / "cldai_renderer.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    # Inject CLD data
    html_with_data = html_template.replace(
        "{{ CLD_JSON_PLACEHOLDER }}",
        json.dumps(cld_json)
    )

    # Render in iframe
    components.html(html_with_data, height=700, scrolling=True)


def generate_standalone_html(cld):
    """Generate standalone HTML file with embedded CLD"""
    from dataclasses import asdict

    cld_json = {
        "metadata": cld.metadata,
        "nodes": [asdict(n) for n in cld.nodes],
        "edges": [asdict(e) for e in cld.edges],
        "loops": [asdict(l) for l in cld.loops],
        "archetypes": cld.archetypes,
        "aha_paradox": cld.aha_paradox
    }

    template_path = Path(__file__).parent / "templates" / "cldai_renderer.html"
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()

    return html_template.replace(
        "{{ CLD_JSON_PLACEHOLDER }}",
        json.dumps(cld_json)
    )


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**CLD Studio v1.0**")
st.sidebar.markdown("*Powered by CAMS Framework & Perplexity AI*")
st.sidebar.caption(f"Model: {model} | Domain: {collision_domain}")
