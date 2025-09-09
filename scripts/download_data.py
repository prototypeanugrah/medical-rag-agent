#!/usr/bin/env python3
"""
Download Medical Data Files from Google Drive

This script helps users download the required medical data files from Google Drive.
The files are hosted at: https://drive.google.com/drive/folders/17xzfvf0njVT7k9TDF6m2mCtDM4vMmV4F?usp=sharing

Usage:
    python scripts/download_data.py --check    # Check which files exist
    python scripts/download_data.py --info     # Show download instructions
"""

import argparse
import os
import webbrowser
from pathlib import Path
from typing import Dict


class DataHelper:
    def __init__(self):
        # Google Drive folder URL
        self.drive_folder_url = "https://drive.google.com/drive/folders/17xzfvf0njVT7k9TDF6m2mCtDM4vMmV4F?usp=sharing"
        
        # Expected data files
        self.data_files = {
            "db_kg.csv": {
                "description": "Knowledge graph relations (1.4M+ records)",
                "size_mb": 200
            },
            "db_meta.txt": {
                "description": "Drug metadata and classifications",
                "size_mb": 5
            },
            "db_food_interactions.txt": {
                "description": "Drug-food interaction warnings", 
                "size_mb": 2
            },
            "db_product.txt": {
                "description": "Drug dosage and formulation data",
                "size_mb": 30
            },
            "db_product_stage.txt": {
                "description": "Market availability and stages",
                "size_mb": 1
            },
            "db_group_mapping.json": {
                "description": "Stage descriptions and definitions",
                "size_mb": 0.001
            }
        }
    
    def show_download_instructions(self) -> None:
        """Show instructions for downloading data from Google Drive"""
        print("📋 Medical RAG Agent - Data Download Instructions")
        print("=" * 50)
        print()
        
        print("🔗 Google Drive Folder:")
        print(f"   {self.drive_folder_url}")
        print()
        
        print("📥 Download Steps:")
        print("1. Click the Google Drive link above (or run with --open)")
        print("2. Select all files in the folder")
        print("3. Right-click and choose 'Download'")
        print("4. Extract the downloaded ZIP file to this project directory")
        print()
        
        print("📁 Required Files:")
        total_size = sum(info["size_mb"] for info in self.data_files.values())
        print(f"   Total size: ~{total_size:.1f}MB")
        print()
        
        for filename, info in self.data_files.items():
            print(f"   ✓ {filename:<25} - {info['description']}")
        print()
        
        print("🚀 After Download:")
        print("   npm run data:check      # Verify files are in place")
        print("   npm run pg:add          # Ingest data into PostgreSQL")
        print("   npm run pg:embeddings   # Generate vector embeddings")
        print()
    
    def check_files(self) -> Dict[str, bool]:
        """Check which files already exist locally"""
        print("📂 Checking for medical data files...")
        print()
        
        status = {}
        found_files = 0
        total_size = 0
        
        for filename, info in self.data_files.items():
            exists = os.path.exists(filename)
            status[filename] = exists
            
            if exists:
                actual_size = os.path.getsize(filename) / 1024 / 1024
                total_size += actual_size
                found_files += 1
                print(f"✅ {filename:<25} ({actual_size:.1f}MB) - {info['description']}")
            else:
                print(f"❌ {filename:<25} - Missing")
        
        print()
        print(f"📊 Summary: {found_files}/{len(self.data_files)} files found ({total_size:.1f}MB)")
        
        if found_files == len(self.data_files):
            print("🎉 All data files are ready!")
            print("🚀 Run 'npm run pg:add' to ingest the data")
        elif found_files > 0:
            print("⚠️  Some files are missing. Please download the remaining files.")
            print("💡 Run with --info to see download instructions")
        else:
            print("📥 No data files found. Please download them first.")
            print("💡 Run with --info to see download instructions")
        
        print()
        return status
    
    def open_drive_folder(self) -> None:
        """Open the Google Drive folder in the default browser"""
        print(f"🌐 Opening Google Drive folder in browser...")
        print(f"📂 {self.drive_folder_url}")
        
        try:
            webbrowser.open(self.drive_folder_url)
            print("✅ Opened in browser")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print("💡 Please manually copy and paste the URL above")
    
    def clean_files(self) -> None:
        """Remove downloaded data files"""
        print("🧹 Cleaning downloaded data files...")
        print()
        
        removed_count = 0
        for filename in self.data_files.keys():
            if os.path.exists(filename):
                try:
                    file_size = os.path.getsize(filename) / 1024 / 1024
                    os.remove(filename)
                    print(f"🗑️  Removed {filename} ({file_size:.1f}MB)")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {filename}: {e}")
            else:
                print(f"⚪ {filename} - Not found")
        
        print()
        print(f"✅ Cleaned {removed_count} data files")
    
    def quick_setup_guide(self) -> None:
        """Show a quick setup guide"""
        print("🚀 Medical RAG Agent - Quick Setup")
        print("=" * 40)
        print()
        print("Step 1: Download Data")
        print("  npm run data:info       # Show download instructions")
        print("  npm run data:open       # Open Google Drive folder")
        print("  # Download all files to project root")
        print()
        print("Step 2: Verify Data")
        print("  npm run data:check      # Check downloaded files")
        print()
        print("Step 3: Setup Database")
        print("  npm run pg:setup       # Create PostgreSQL tables")
        print("  npm run pg:add         # Ingest medical data (5-10 min)")
        print("  npm run pg:embeddings  # Generate vectors (~$1.76, 4-6 hours)")
        print()
        print("Step 4: Start Application")
        print("  npm run dev            # Start frontend + backend")
        print()


def main():
    parser = argparse.ArgumentParser(description="Medical data download helper")
    parser.add_argument("--check", action="store_true", help="Check which files exist locally")
    parser.add_argument("--info", action="store_true", help="Show download instructions")
    parser.add_argument("--open", action="store_true", help="Open Google Drive folder in browser")
    parser.add_argument("--clean", action="store_true", help="Remove downloaded files")
    parser.add_argument("--guide", action="store_true", help="Show quick setup guide")
    
    args = parser.parse_args()
    
    helper = DataHelper()
    
    if args.check:
        helper.check_files()
    elif args.info:
        helper.show_download_instructions()
    elif args.open:
        helper.open_drive_folder()
    elif args.clean:
        helper.clean_files()
    elif args.guide:
        helper.quick_setup_guide()
    else:
        # Default: show info and check files
        helper.show_download_instructions()
        print()
        helper.check_files()


if __name__ == "__main__":
    main()