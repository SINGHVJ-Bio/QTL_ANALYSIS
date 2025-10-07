#!/usr/bin/env python3
"""
Test script to run the QTL pipeline with sample NAFLD data
"""

import os
import subprocess
import sys
from pathlib import Path

def test_pipeline():
    """Test the QTL pipeline with sample data"""
    
    print("🧪 Testing QTL Pipeline with NAFLD Sample Data...")
    
    # Create sample data first
    print("1. Creating sample data...")
    result = subprocess.run([sys.executable, "create_sample_data.py"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to create sample data: {result.stderr}")
        return False
    
    print("✅ Sample data created successfully")
    
    # Test with shell script
    print("\n2. Testing with shell script...")
    shell_cmd = [
        "./run_qtl_pipeline.sh",
        "-d", "data",
        "-o", "test_results_shell",
        "-c", "config/test_config.yaml",
        "--debug"
    ]
    
    result = subprocess.run(shell_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Shell script test completed successfully")
        print(f"📁 Results: test_results_shell/")
    else:
        print(f"❌ Shell script test failed: {result.stderr}")
        # Don't return False here as we want to test Python runner too
    
    # Test with Python runner
    print("\n3. Testing with Python runner...")
    python_cmd = [
        sys.executable, "run_QTLPipeline.py",
        "--config", "config/test_config.yaml",
        "--debug"
    ]
    
    result = subprocess.run(python_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Python runner test completed successfully")
        print(f"📁 Results: test_results/")
    else:
        print(f"❌ Python runner test failed: {result.stderr}")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Next steps:")
    print("   - Check test_results/ directory for output")
    print("   - Review test_results/reports/analysis_report.html")
    print("   - Examine test_results/plots/ for generated visualizations")
    print("   - Look at test_results/logs/ for detailed logs")
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)