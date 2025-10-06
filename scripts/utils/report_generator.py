#!/usr/bin/env python3
"""
Report generation utilities
"""

import os
import pandas as pd
from datetime import datetime

def generate_html_report(report_data, output_file):
    """Generate comprehensive HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QTL Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .success {{ color: green; font-weight: bold; }}
            .failed {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .plot-item {{ text-align: center; }}
            .plot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>QTL Analysis Report</h1>
            <p><strong>Generated:</strong> {report_data['timestamp']}</p>
            <p><strong>Runtime:</strong> {report_data['runtime']}</p>
            <p><strong>Results Directory:</strong> {report_data['config']['results_dir']}</p>
        </div>
        
        {generate_analysis_section(report_data)}
        {generate_plot_section(report_data)}
        {generate_config_section(report_data)}
        
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def generate_analysis_section(report_data):
    """Generate analysis results section"""
    sections = []
    
    # QTL Analysis
    if 'qtl' in report_data['results']:
        qtl_html = """
        <div class="section">
            <h2>QTL Analysis Results</h2>
            <table>
                <tr><th>Analysis Type</th><th>Status</th><th>Significant Associations</th><th>Result File</th></tr>
        """
        
        for qtl_type, result in report_data['results']['qtl'].items():
            status_class = "success" if result['status'] == 'completed' else "failed"
            status_text = "COMPLETED" if result['status'] == 'completed' else "FAILED"
            count = result.get('significant_count', 0) if result['status'] == 'completed' else 'N/A'
            result_file = os.path.basename(result.get('result_file', 'N/A'))
            
            qtl_html += f"""
                <tr>
                    <td>{qtl_type.upper()}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{count}</td>
                    <td>{result_file}</td>
                </tr>
            """
        
        qtl_html += "</table></div>"
        sections.append(qtl_html)
    
    # GWAS Analysis
    if 'gwas' in report_data['results']:
        gwas_result = report_data['results']['gwas']
        status_class = "success" if gwas_result['status'] == 'completed' else "failed"
        status_text = "COMPLETED" if gwas_result['status'] == 'completed' else "FAILED"
        count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
        method = gwas_result.get('method', 'N/A')
        
        gwas_html = f"""
        <div class="section">
            <h2>GWAS Analysis Results</h2>
            <table>
                <tr><th>Analysis Type</th><th>Status</th><th>Method</th><th>Significant Associations</th></tr>
                <tr>
                    <td>GWAS</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{method}</td>
                    <td>{count}</td>
                </tr>
            </table>
        </div>
        """
        sections.append(gwas_html)
    
    return "\n".join(sections)

def generate_plot_section(report_data):
    """Generate plot section with embedded images"""
    plots_dir = os.path.join(report_data['config']['results_dir'], "plots")
    if not os.path.exists(plots_dir):
        return "<div class='section'><h2>Plots</h2><p>No plots generated.</p></div>"
    
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.svg'))]
    if not plot_files:
        return "<div class='section'><h2>Plots</h2><p>No plots available.</p></div>"
    
    plot_html = "<div class='section'><h2>Generated Plots</h2><div class='plot-grid'>"
    
    for plot_file in sorted(plot_files):
        plot_path = os.path.join("plots", plot_file)
        plot_name = os.path.splitext(plot_file)[0].replace('_', ' ').title()
        plot_html += f"""
            <div class="plot-item">
                <img src="{plot_path}" alt="{plot_name}">
                <p><strong>{plot_name}</strong></p>
            </div>
        """
    
    plot_html += "</div></div>"
    return plot_html

def generate_config_section(report_data):
    """Generate configuration summary section"""
    config = report_data['config']
    
    config_html = """
    <div class="section">
        <h2>Configuration Summary</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    # Main configuration
    config_html += f"<tr><td>Results Directory</td><td>{config['results_dir']}</td></tr>"
    config_html += f"<tr><td>QTL Types</td><td>{config['analysis']['qtl_types']}</td></tr>"
    config_html += f"<tr><td>GWAS Analysis</td><td>{config['analysis'].get('run_gwas', False)}</td></tr>"
    
    # QTL parameters
    if 'qtl' in config:
        qtl_config = config['qtl']
        config_html += f"<tr><td>Cis Window</td><td>{qtl_config.get('cis_window', 'N/A')}</td></tr>"
        config_html += f"<tr><td>Permutations</td><td>{qtl_config.get('permutations', 'N/A')}</td></tr>"
        config_html += f"<tr><td>FDR Threshold</td><td>{qtl_config.get('fdr_threshold', 'N/A')}</td></tr>"
    
    config_html += "</table></div>"
    return config_html

def generate_summary_report(report_data, output_file):
    """Generate text summary report"""
    with open(output_file, 'w') as f:
        f.write("QTL ANALYSIS PIPELINE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {report_data['timestamp']}\n")
        f.write(f"Runtime: {report_data['runtime']}\n")
        f.write(f"Results Directory: {report_data['config']['results_dir']}\n\n")
        
        f.write("ANALYSIS RESULTS\n")
        f.write("-" * 50 + "\n\n")
        
        # QTL Results
        if 'qtl' in report_data['results']:
            f.write("QTL Analysis:\n")
            for qtl_type, result in report_data['results']['qtl'].items():
                status = "COMPLETED" if result['status'] == 'completed' else "FAILED"
                count = result.get('significant_count', 0) if result['status'] == 'completed' else 'N/A'
                f.write(f"  {qtl_type.upper():<8} {status:<12} Significant: {count}\n")
            f.write("\n")
        
        # GWAS Results
        if 'gwas' in report_data['results']:
            gwas_result = report_data['results']['gwas']
            status = "COMPLETED" if gwas_result['status'] == 'completed' else "FAILED"
            count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
            method = gwas_result.get('method', 'N/A')
            f.write(f"GWAS Analysis: {status} (Method: {method}, Significant: {count})\n\n")
        
        f.write("OUTPUT DIRECTORY STRUCTURE\n")
        f.write("-" * 50 + "\n")
        f.write("results/\n")
        f.write("├── qtl_results/     # QTL analysis results\n")
        f.write("├── gwas_results/    # GWAS analysis results\n")
        f.write("├── plots/           # Generated plots\n")
        f.write("├── reports/         # Analysis reports\n")
        f.write("├── logs/            # Pipeline logs\n")
        f.write("└── temp/            # Temporary files\n")