"""Fix syntax error in cams_realtime_monitor.py"""

# Read the file
with open('cams_realtime_monitor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the problematic line with embedded newline characters
problematic_line = 'create_alert_banner(country_data, selected_country)\\n            \\n            # Thermodynamic Metrics Dashboard\\n            if analysis.get(\'thermodynamic_metrics\'):\\n                st.markdown("### 🔬 CAMS-THERMO System Analysis")\\n                thermo = analysis[\'thermodynamic_metrics\']\\n                \\n                col1, col2, col3, col4 = st.columns(4)\\n                \\n                with col1:\\n                    st.metric(\\n                        "Total Energy", \\n                        f"{thermo[\'total_energy\']:.1f}",\\n                        help="Sum of all node energies (α·C² + β·K - γ·S)"\\n                    )\\n                \\n                with col2:\\n                    st.metric(\\n                        "System Entropy", \\n                        f"{thermo[\'total_dissipation\']:.1f}",\\n                        help="Total energy dissipation (δ·S·ln(1+A))"\\n                    )\\n                \\n                with col3:\\n                    st.metric(\\n                        "Free Energy", \\n                        f"{thermo[\'total_free_energy\']:.1f}",\\n                        help="Available energy for work ((K-S)·(1-A/10))"\\n                    )\\n                \\n                with col4:\\n                    efficiency = thermo[\'efficiency\']\\n                    st.metric(\\n                        "Efficiency", \\n                        f"{efficiency:.2f}",\\n                        help="Free Energy / Dissipation ratio"\\n                    )\\n                \\n                # Heat sink indicators\\n                if thermo[\'heat_sinks\'] > 0:\\n                    st.warning(f"⚠️ {thermo[\'heat_sinks\']} node(s) acting as heat sinks (K < S and A < 5)")\\n                else:\\n                    st.success("✅ No heat sink nodes detected - system energy flow is stable")\\n                \\n                # Thermodynamic health interpretation\\n                if efficiency > 2.0:\\n                    thermo_status = "🟢 Thermodynamically Stable"\\n                elif efficiency > 1.0:\\n                    thermo_status = "🟡 Moderate Efficiency"\\n                elif efficiency > 0.5:\\n                    thermo_status = "🟠 Low Efficiency - Energy Waste"\\n                else:\\n                    thermo_status = "🔴 Critical - High Dissipation"\\n                \\n                st.info(f"**Thermodynamic Status:** {thermo_status}")'

fixed_line = '''create_alert_banner(country_data, selected_country)
            
            # Thermodynamic Metrics Dashboard
            if analysis.get('thermodynamic_metrics'):
                st.markdown("### 🔬 CAMS-THERMO System Analysis")
                thermo = analysis['thermodynamic_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Energy", 
                        f"{thermo['total_energy']:.1f}",
                        help="Sum of all node energies (α·C² + β·K - γ·S)"
                    )
                
                with col2:
                    st.metric(
                        "System Entropy", 
                        f"{thermo['total_dissipation']:.1f}",
                        help="Total energy dissipation (δ·S·ln(1+A))"
                    )
                
                with col3:
                    st.metric(
                        "Free Energy", 
                        f"{thermo['total_free_energy']:.1f}",
                        help="Available energy for work ((K-S)·(1-A/10))"
                    )
                
                with col4:
                    efficiency = thermo['efficiency']
                    st.metric(
                        "Efficiency", 
                        f"{efficiency:.2f}",
                        help="Free Energy / Dissipation ratio"
                    )
                
                # Heat sink indicators
                if thermo['heat_sinks'] > 0:
                    st.warning(f"⚠️ {thermo['heat_sinks']} node(s) acting as heat sinks (K < S and A < 5)")
                else:
                    st.success("✅ No heat sink nodes detected - system energy flow is stable")
                
                # Thermodynamic health interpretation
                if efficiency > 2.0:
                    thermo_status = "🟢 Thermodynamically Stable"
                elif efficiency > 1.0:
                    thermo_status = "🟡 Moderate Efficiency"
                elif efficiency > 0.5:
                    thermo_status = "🟠 Low Efficiency - Energy Waste"
                else:
                    thermo_status = "🔴 Critical - High Dissipation"
                
                st.info(f"**Thermodynamic Status:** {thermo_status}")'''

# Replace the problematic section
if problematic_line in content:
    content = content.replace(problematic_line, fixed_line)
    print("Fixed the syntax error!")
else:
    print("Problematic line not found, trying alternative approach...")
    # Try to find and fix any embedded newline characters
    content = content.replace('\\n', '\n')
    print("Fixed embedded newline characters")

# Write the corrected content back
with open('cams_realtime_monitor.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("File syntax error fixed successfully!")