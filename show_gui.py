#!/usr/bin/env python3
"""
Simple script to show/focus the ChessLab GUI window
"""

import subprocess
import time
import os

def bring_gui_to_front():
    """Try to bring the ChessLab GUI to the front"""
    try:
        # Try to focus on Python windows (macOS)
        subprocess.run([
            'osascript', '-e', 
            'tell application "System Events" to set frontmost of every process whose name contains "Python" to true'
        ], check=False)
        
        print("✅ Attempted to bring ChessLab GUI to front")
        
        # Also try to focus the terminal to see this message
        subprocess.run([
            'osascript', '-e',
            'tell application "Terminal" to activate'
        ], check=False)
        
    except Exception as e:
        print(f"⚠️  Could not bring GUI to front: {e}")

def check_gui_status():
    """Check if GUI is running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'chess_lab_gui.py' in result.stdout:
            print("✅ ChessLab GUI is running")
            return True
        else:
            print("❌ ChessLab GUI is not running")
            return False
    except:
        return False

def main():
    print("🔍 ChessLab GUI Status Check")
    print("=" * 30)
    
    if check_gui_status():
        print("\n🖥️  Attempting to show GUI window...")
        bring_gui_to_front()
        
        print("\n💡 If you still don't see the GUI:")
        print("   1. Look for 'ChessLab - Advanced Chess Evaluation' window")
        print("   2. Press Cmd+Tab to cycle through applications")
        print("   3. Check your dock for Python/ChessLab icon")
        print("   4. Try moving/clicking around your screen")
        
        print("\n🎮 GUI Features Available:")
        print("   • Stockfish Depth 0-15 (~800-3200+ ELO)")
        print("   • LC0 100-1600 nodes (~1500-2800 ELO)")
        print("   • Human Player support")
        print("   • Auto-play tournaments")
        print("   • Position evaluation display")
        print("   • Our 64M Model (when iteration 8 completes)")
        
    else:
        print("\n🚀 Starting ChessLab GUI...")
        os.system("python chess_lab_gui.py &")
        time.sleep(2)
        check_gui_status()

if __name__ == "__main__":
    main() 