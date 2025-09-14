#!/usr/bin/env python3
"""
Training Data Bot - Cleanup Utility

Simple script to clean up temporary files, empty datasets, and organize the workspace.
"""

import os
import sys
from pathlib import Path
import shutil
import json
from typing import List


class WorkspaceCleanup:
    """Workspace cleanup utility."""
    
    def __init__(self, workspace_root: Path = None):
        self.root = workspace_root or Path.cwd()
        self.cleaned_files = []
        self.preserved_files = []
        
    def clean_empty_jsonl_files(self) -> None:
        """Remove empty JSONL files."""
        print("🧹 Cleaning empty JSONL files...")
        
        jsonl_files = list(self.root.rglob("*.jsonl"))
        for file_path in jsonl_files:
            if file_path.stat().st_size == 0:
                file_path.unlink()
                self.cleaned_files.append(str(file_path))
                print(f"   ❌ Removed: {file_path}")
            else:
                self.preserved_files.append(str(file_path))
                
    def clean_empty_directories(self) -> None:
        """Remove empty directories."""
        print("🧹 Cleaning empty directories...")
        
        # Common directories that should exist but can be empty
        preserve_dirs = {"cache", "temp", "data", "output", "outputs"}
        
        for dirpath in sorted(self.root.rglob("*"), reverse=True):
            if dirpath.is_dir() and dirpath.name not in preserve_dirs:
                try:
                    if not any(dirpath.iterdir()):
                        dirpath.rmdir()
                        self.cleaned_files.append(str(dirpath))
                        print(f"   ❌ Removed empty dir: {dirpath}")
                except OSError:
                    pass  # Directory not empty or permission denied
                    
    def clean_python_cache(self) -> None:
        """Remove Python cache files."""
        print("🧹 Cleaning Python cache...")
        
        cache_patterns = ["__pycache__", "*.pyc", "*.pyo", ".pytest_cache"]
        
        for pattern in cache_patterns:
            for cache_path in self.root.rglob(pattern):
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                    self.cleaned_files.append(str(cache_path))
                    print(f"   ❌ Removed: {cache_path}")
                elif cache_path.is_file():
                    cache_path.unlink()
                    self.cleaned_files.append(str(cache_path))
                    print(f"   ❌ Removed: {cache_path}")
                    
    def clean_test_files(self) -> None:
        """Remove temporary test files."""
        print("🧹 Cleaning test files...")
        
        test_patterns = [
            "test_*.py",
            "*_test.py", 
            "simple_*.py",
            "example_*.py",
            "temp_*.py"
        ]
        
        for pattern in test_patterns:
            for test_file in self.root.rglob(pattern):
                # Only remove if file is small (likely a temp test)
                if test_file.stat().st_size < 1000:
                    try:
                        content = test_file.read_text()
                        # Remove if mostly empty or contains test patterns
                        if len(content.strip()) < 50 or "# temp" in content.lower():
                            test_file.unlink()
                            self.cleaned_files.append(str(test_file))
                            print(f"   ❌ Removed: {test_file}")
                    except Exception:
                        pass  # Skip if can't read
                        
    def organize_output_files(self) -> None:
        """Organize output files into proper directories."""
        print("📁 Organizing output files...")
        
        # Ensure output directory exists
        output_dir = self.root / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Move scattered JSONL files to output directory
        for jsonl_file in self.root.glob("*.jsonl"):
            if jsonl_file.parent == self.root:  # Only files in root
                target = output_dir / jsonl_file.name
                if not target.exists():
                    shutil.move(str(jsonl_file), str(target))
                    print(f"   📦 Moved: {jsonl_file} → {target}")
                    
    def validate_datasets(self) -> None:
        """Validate and report on dataset files."""
        print("✅ Validating datasets...")
        
        datasets = list(self.root.rglob("*.jsonl"))
        
        for dataset in datasets:
            try:
                if dataset.stat().st_size > 0:
                    with open(dataset, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Try to parse first line as JSON
                            json.loads(lines[0])
                            print(f"   ✅ Valid dataset: {dataset} ({len(lines)} examples)")
                        else:
                            print(f"   ⚠️  Empty dataset: {dataset}")
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON in: {dataset}")
            except Exception as e:
                print(f"   ⚠️  Could not validate: {dataset} ({e})")
                
    def create_gitignore(self) -> None:
        """Create/update .gitignore file."""
        print("📝 Updating .gitignore...")
        
        gitignore_content = """
# Training Data Bot - Generated Files
*.pyc
__pycache__/
.pytest_cache/
.coverage
htmlcov/

# Data files
output/*.jsonl
cache/
temp/
*.log

# Environment
.env
.env.local
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model files
*.bin
*.safetensors
models/

# Large datasets
data/large/
datasets/
*.csv
*.tsv
""".strip()

        gitignore_path = self.root / ".gitignore"
        
        if not gitignore_path.exists():
            gitignore_path.write_text(gitignore_content)
            print(f"   ✅ Created: {gitignore_path}")
        else:
            existing = gitignore_path.read_text()
            if "# Training Data Bot" not in existing:
                with open(gitignore_path, 'a') as f:
                    f.write(f"\n\n{gitignore_content}")
                print(f"   ✅ Updated: {gitignore_path}")
                
    def run_full_cleanup(self) -> None:
        """Run all cleanup operations."""
        print("🚀 Starting workspace cleanup...\n")
        
        self.clean_empty_jsonl_files()
        self.clean_empty_directories()
        self.clean_python_cache()
        self.clean_test_files()
        self.organize_output_files()
        self.validate_datasets()
        self.create_gitignore()
        
        print(f"\n✨ Cleanup complete!")
        print(f"   📁 Files cleaned: {len(self.cleaned_files)}")
        print(f"   💾 Files preserved: {len(self.preserved_files)}")
        
        if self.cleaned_files:
            print("\n🗑️  Cleaned files:")
            for file in self.cleaned_files[:10]:  # Show first 10
                print(f"   - {file}")
            if len(self.cleaned_files) > 10:
                print(f"   ... and {len(self.cleaned_files) - 10} more")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("""
Training Data Bot Cleanup Utility

Usage:
    python cleanup.py              # Run full cleanup
    python cleanup.py --dry-run    # Show what would be cleaned
    
Operations:
    - Remove empty JSONL files
    - Clean Python cache files  
    - Remove temporary test files
    - Organize output files
    - Validate datasets
    - Update .gitignore
        """)
        return
        
    dry_run = len(sys.argv) > 1 and sys.argv[1] == "--dry-run"
    
    if dry_run:
        print("🔍 DRY RUN - No files will be modified\n")
    
    cleanup = WorkspaceCleanup()
    
    if not dry_run:
        cleanup.run_full_cleanup()
    else:
        print("Would clean Python cache, empty files, and organize outputs.")
        print("Run without --dry-run to perform actual cleanup.")


if __name__ == "__main__":
    main() 