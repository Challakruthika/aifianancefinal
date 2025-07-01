#!/usr/bin/env python3
"""
Quick Demo Launcher for AI-Enhanced Financial Dashboard
Opens the HTML demo automatically in your browser
"""

import webbrowser
import time
import subprocess
import sys
import os

def launch_demo():
    print("🚀 Launching AI-Enhanced Financial Dashboard Demo...")
    print("")
    
    # Check if simple_demo.html exists
    demo_path = "simple_demo.html"
    if not os.path.exists(demo_path):
        print("❌ Demo file not found!")
        return
    
    print("✅ Demo file found!")
    print("🌐 Starting local server...")
    
    # Get the full path to the demo file
    full_path = os.path.abspath(demo_path)
    file_url = f"file://{full_path}"
    
    print(f"📱 Opening: {file_url}")
    print("")
    print("🎯 DEMO FEATURES:")
    print("   • Beautiful AI dashboard interface")
    print("   • Interactive charts and visualizations")
    print("   • 5 different analysis sections")
    print("   • Working AI chatbot")
    print("   • Real financial data analysis")
    print("")
    
    # Try to open in browser
    try:
        webbrowser.open(file_url)
        print("✅ Demo opened in your default browser!")
        print("")
        print("🎮 NAVIGATION:")
        print("   🏠 Portfolio Overview - Start here")
        print("   🤖 AI Risk Analytics - ML predictions")
        print("   🔍 Anomaly Detection - Fraud analysis")
        print("   💬 AI Advisor - Chat with AI")
        print("   🔮 Predictions - Future forecasts")
        print("")
        print("💡 If browser doesn't open automatically:")
        print(f"   Copy and paste: {file_url}")
        
    except Exception as e:
        print("❌ Could not open browser automatically")
        print(f"📋 Manual link: {file_url}")
        print("   Copy and paste this link into your browser")
    
    print("")
    print("🎉 Enjoy exploring your AI-Enhanced Financial Dashboard!")

if __name__ == "__main__":
    launch_demo()