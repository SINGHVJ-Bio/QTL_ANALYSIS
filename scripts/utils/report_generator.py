#!/usr/bin/env python3
"""
Enhanced report generation utilities with comprehensive analysis summaries
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

def generate_html_report(report_data, output_file):
    """Generate comprehensive HTML report with detailed analysis results"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QTL Analysis Report</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                line-height: 1.6; 
                color: #333;
                background-color: #f8f9fa;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{ 
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                padding: 30px;
                border-radius: 8px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{ 
                margin: 0; 
                font-size: 2.5em;
                font-weight: 300;
            }}
            .header p {{ 
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .section {{ 
                margin: 25px 0; 
                padding: 20px;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                background: #fff;
            }}
            .section h2 {{ 
                color: #2E86AB;
                border-bottom: 2px solid #F18F01;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .success {{ color: #28a745; font-weight: bold; }}
            .failed {{ color: #dc3545; font-weight: bold; }}
            .warning {{ color: #ffc107; font-weight: bold; }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 15px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            th, td {{ 
                border: 1px solid #dee2e6; 
                padding: 12px 15px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #2E86AB; 
                color: white;
                font-weight: 600;
            }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #e9ecef; }}
            .plot-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin: 20px 0; 
            }}
            .plot-item {{ 
                text-align: center; 
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .plot-item img {{ 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #ddd; 
                border-radius: 5px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .summary-box {{
                background: #e7f3ff;
                border-left: 4px solid #2E86AB;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }}
            .code {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #F18F01;
                font-family: 'Courier New', monospace;
                margin: 15px 0;
            }}
            @media (max-width: 768px) {{
                .container {{ padding: 15px; }}
                .plot-grid {{ grid-template-columns: 1fr; }}
                .stats-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 QTL Analysis Report</h1>
                <p><strong>Comprehensive Analysis of Genetic Regulation</strong></p>
                <p>Generated: {report_data['timestamp']} | Runtime: {report_data['runtime']}</p>
            </div>
            
            {generate_executive_summary(report_data)}
            {generate_analysis_results_section(report_data)}
            {generate_plot_section(report_data)}
            {generate_configuration_section(report_data)}
            {generate_next_steps_section(report_data)}
            
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def generate_executive_summary(report_data):
    """Generate executive summary section"""
    total_significant = 0
    completed_analyses = 0
    
    # Count significant associations and completed analyses
    if 'qtl' in report_data['results']:
        for qtl_type, result in report_data['results']['qtl'].items():
            if 'cis' in result and result['cis']['status'] == 'completed':
                total_significant += result['cis'].get('significant_count', 0)
                completed_analyses += 1
            if 'trans' in result and result['trans']['status'] == 'completed':
                total_significant += result['trans'].get('significant_count', 0)
                completed_analyses += 1
    
    if 'gwas' in report_data['results'] and report_data['results']['gwas']['status'] == 'completed':
        total_significant += report_data['results']['gwas'].get('significant_count', 0)
        completed_analyses += 1
    
    return f"""
    <div class="section">
        <h2>📊 Executive Summary</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Analyses</div>
                <div class="stat-number">{completed_analyses}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Significant</div>
                <div class="stat-number">{total_significant}</div>
                <div class="stat-label">Associations</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Results Directory</div>
                <div class="stat-number" style="font-size: 1.2em;">{os.path.basename(report_data['results_dir'])}</div>
                <div class="stat-label">Output Location</div>
            </div>
        </div>
        
        <div class="summary-box">
            <strong>🎯 Key Findings:</strong>
            <ul>
                <li>Analysis completed successfully in {report_data['runtime']}</li>
                <li>Found {total_significant} significant genetic associations</li>
                <li>Comprehensive results available in the sections below</li>
                <li>All generated plots and data files are available for download</li>
            </ul>
        </div>
    </div>
    """

def generate_analysis_results_section(report_data):
    """Generate detailed analysis results section"""
    sections = []
    
    # QTL Analysis Results
    if 'qtl' in report_data['results']:
        qtl_html = """
        <div class="section">
            <h2>🔬 QTL Analysis Results</h2>
            <table>
                <tr>
                    <th>Analysis Type</th>
                    <th>Mode</th>
                    <th>Status</th>
                    <th>Significant Associations</th>
                    <th>Result File</th>
                </tr>
        """
        
        for qtl_type, result in report_data['results']['qtl'].items():
            # CIS results
            if 'cis' in result:
                cis_result = result['cis']
                status_class = "success" if cis_result['status'] == 'completed' else "failed"
                status_text = "✅ COMPLETED" if cis_result['status'] == 'completed' else "❌ FAILED"
                count = cis_result.get('significant_count', 0) if cis_result['status'] == 'completed' else 'N/A'
                result_file = os.path.basename(cis_result.get('result_file', 'N/A'))
                
                qtl_html += f"""
                <tr>
                    <td>{qtl_type.upper()}</td>
                    <td>CIS</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{count}</td>
                    <td>{result_file}</td>
                </tr>
                """
            
            # TRANS results
            if 'trans' in result:
                trans_result = result['trans']
                status_class = "success" if trans_result['status'] == 'completed' else "failed"
                status_text = "✅ COMPLETED" if trans_result['status'] == 'completed' else "❌ FAILED"
                count = trans_result.get('significant_count', 0) if trans_result['status'] == 'completed' else 'N/A'
                result_file = os.path.basename(trans_result.get('result_file', 'N/A'))
                
                qtl_html += f"""
                <tr>
                    <td>{qtl_type.upper()}</td>
                    <td>TRANS</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{count}</td>
                    <td>{result_file}</td>
                </tr>
                """
        
        qtl_html += "</table></div>"
        sections.append(qtl_html)
    
    # GWAS Analysis Results
    if 'gwas' in report_data['results']:
        gwas_result = report_data['results']['gwas']
        status_class = "success" if gwas_result['status'] == 'completed' else "failed"
        status_text = "✅ COMPLETED" if gwas_result['status'] == 'completed' else "❌ FAILED"
        count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
        method = gwas_result.get('method', 'N/A')
        
        gwas_html = f"""
        <div class="section">
            <h2>📈 GWAS Analysis Results</h2>
            <table>
                <tr>
                    <th>Analysis Type</th>
                    <th>Status</th>
                    <th>Method</th>
                    <th>Significant Associations</th>
                    <th>P-value Threshold</th>
                </tr>
                <tr>
                    <td>GWAS</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{method}</td>
                    <td>{count}</td>
                    <td>5e-8</td>
                </tr>
            </table>
        </div>
        """
        sections.append(gwas_html)
    
    return "\n".join(sections)

def generate_plot_section(report_data):
    """Generate plot section with embedded images"""
    plots_dir = os.path.join(report_data['results_dir'], "plots")
    if not os.path.exists(plots_dir):
        return "<div class='section'><h2>📊 Generated Plots</h2><p>No plots were generated for this analysis.</p></div>"
    
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.svg', '.pdf'))]
    if not plot_files:
        return "<div class='section'><h2>📊 Generated Plots</h2><p>No plot files found in the plots directory.</p></div>"
    
    plot_html = "<div class='section'><h2>📊 Generated Plots</h2><div class='plot-grid'>"
    
    for plot_file in sorted(plot_files):
        if plot_file.startswith('.') or 'summary' in plot_file.lower():
            continue
            
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

def generate_configuration_section(report_data):
    """Generate configuration summary section"""
    config = report_data['config']
    
    config_html = """
    <div class="section">
        <h2>⚙️ Configuration Summary</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    # Main configuration
    config_html += f"<tr><td>Results Directory</td><td>{config['results_dir']}</td></tr>"
    config_html += f"<tr><td>QTL Types</td><td>{config['analysis']['qtl_types']}</td></tr>"
    config_html += f"<tr><td>QTL Mode</td><td>{config['analysis'].get('qtl_mode', 'cis')}</td></tr>"
    config_html += f"<tr><td>GWAS Analysis</td><td>{config['analysis'].get('run_gwas', False)}</td></tr>"
    
    # QTL parameters
    if 'qtl' in config:
        qtl_config = config['qtl']
        config_html += f"<tr><td>Cis Window</td><td>{qtl_config.get('cis_window', 'N/A')} bp</td></tr>"
        config_html += f"<tr><td>Permutations</td><td>{qtl_config.get('permutations', 'N/A')}</td></tr>"
        config_html += f"<tr><td>FDR Threshold</td><td>{qtl_config.get('fdr_threshold', 'N/A')}</td></tr>"
        config_html += f"<tr><td>MAF Threshold</td><td>{qtl_config.get('maf_threshold', 'N/A')}</td></tr>"
    
    # Input files
    config_html += "<tr><td colspan='2' style='background-color: #e9ecef; text-align: center;'><strong>Input Files</strong></td></tr>"
    for file_type, file_path in config['input_files'].items():
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            config_html += f"<tr><td>{file_type.title()}</td><td>{file_path} ({file_size:.1f} MB)</td></tr>"
        elif file_path:
            config_html += f"<tr><td>{file_type.title()}</td><td>{file_path} <span class='warning'>(not found)</span></td></tr>"
    
    config_html += "</table></div>"
    return config_html

def generate_next_steps_section(report_data):
    """Generate next steps and recommendations section"""
    return """
    <div class="section">
        <h2>🎯 Next Steps & Recommendations</h2>
        
        <div class="summary-box">
            <strong>📋 Immediate Actions:</strong>
            <ul>
                <li>Review the significant associations in the results directory</li>
                <li>Examine generated plots for quality control and results visualization</li>
                <li>Check the pipeline logs for any warnings or additional information</li>
            </ul>
        </div>
        
        <div class="summary-box">
            <strong>🔬 Further Analysis:</strong>
            <ul>
                <li>Consider functional annotation of significant variants</li>
                <li>Perform pathway enrichment analysis on associated genes</li>
                <li>Validate top hits in independent datasets if available</li>
                <li>Explore conditional analysis to identify independent signals</li>
            </ul>
        </div>
        
        <div class="code">
            # Example command to view top associations<br>
            head -20 QTL_results/eqtl_cis_significant.txt<br><br>
            
            # Example command to create additional visualizations<br>
            # (Custom R or Python scripts can be used for advanced plotting)
        </div>
    </div>
    """

def generate_summary_report(report_data, output_file):
    """Generate comprehensive text summary report"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QTL ANALYSIS PIPELINE - COMPREHENSIVE SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {report_data['timestamp']}\n")
        f.write(f"Runtime: {report_data['runtime']}\n")
        f.write(f"Results Directory: {report_data['results_dir']}\n\n")
        
        f.write("ANALYSIS RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        
        # QTL Results
        if 'qtl' in report_data['results']:
            f.write("QTL ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            for qtl_type, result in report_data['results']['qtl'].items():
                if 'cis' in result:
                    cis_status = "COMPLETED" if result['cis']['status'] == 'completed' else "FAILED"
                    cis_count = result['cis'].get('significant_count', 0) if result['cis']['status'] == 'completed' else 'N/A'
                    f.write(f"  {qtl_type.upper():<8} CIS:  {cis_status:<12} Significant: {cis_count}\n")
                
                if 'trans' in result:
                    trans_status = "COMPLETED" if result['trans']['status'] == 'completed' else "FAILED"
                    trans_count = result['trans'].get('significant_count', 0) if result['trans']['status'] == 'completed' else 'N/A'
                    f.write(f"  {qtl_type.upper():<8} TRANS: {trans_status:<12} Significant: {trans_count}\n")
            f.write("\n")
        
        # GWAS Results
        if 'gwas' in report_data['results']:
            gwas_result = report_data['results']['gwas']
            status = "COMPLETED" if gwas_result['status'] == 'completed' else "FAILED"
            count = gwas_result.get('significant_count', 0) if gwas_result['status'] == 'completed' else 'N/A'
            method = gwas_result.get('method', 'N/A')
            f.write(f"GWAS ANALYSIS: {status} (Method: {method}, Significant: {count})\n\n")
        
        f.write("OUTPUT DIRECTORY STRUCTURE\n")
        f.write("-" * 80 + "\n")
        f.write(f"{report_data['results_dir']}/\n")
        f.write("├── QTL_results/          # QTL analysis results\n")
        f.write("│   ├── eqtl_cis_significant.txt\n")
        f.write("│   ├── eqtl_cis_nominals.txt\n")
        f.write("│   ├── pqtl_cis_significant.txt\n")
        f.write("│   └── ...\n")
        f.write("├── GWAS_results/         # GWAS analysis results\n")
        f.write("├── plots/                # Generated visualizations\n")
        f.write("├── reports/              # Analysis reports\n")
        f.write("├── logs/                 # Pipeline execution logs\n")
        f.write("├── genotype_processing/  # Processed genotype files\n")
        f.write("└── QC_reports/           # Quality control reports\n\n")
        
        f.write("KEY FILES:\n")
        f.write("-" * 40 + "\n")
        f.write("📊 analysis_report.html    - Comprehensive HTML report\n")
        f.write("📋 pipeline_summary.txt    - This summary file\n")
        f.write("📈 results_metadata.json   - Analysis metadata\n")
        f.write("🖼️  plots/                 - All generated visualizations\n")
        f.write("📝 logs/                   - Detailed execution logs\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Review significant associations in QTL_results/\n")
        f.write("2. Examine plots for quality control and results\n")
        f.write("3. Check logs for any warnings or additional info\n")
        f.write("4. Consider functional follow-up of top hits\n")
        f.write("5. Validate findings in independent datasets\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS COMPLETED SUCCESSFULLY\n")
        f.write("=" * 80 + "\n")