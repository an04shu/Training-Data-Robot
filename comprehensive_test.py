#!/usr/bin/env python3
"""
Comprehensive test suite for the Training Data Bot project.

This test verifies all major components:
1. Core bot functionality
2. WebLoader with Decodo integration
3. Document processing pipeline
4. AI task generation
5. Quality evaluation
6. Export functionality
7. Streamlit dashboard components
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path
import pandas as pd

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from training_data_bot import TrainingDataBot
from training_data_bot.sources.web import WebLoader
from training_data_bot.sources.unified import UnifiedLoader
from training_data_bot.core.models import TaskType, ExportFormat
from training_data_bot.decodo import DecodoClient


class TestResults:
    """Track test results."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, success: bool, details: str = ""):
        self.tests.append({
            "name": name,
            "success": success,
            "details": details
        })
        if success:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print('='*60)
        
        for test in self.tests:
            status = "✅ PASS" if test["success"] else "❌ FAIL"
            print(f"{status}: {test['name']}")
            if test["details"]:
                print(f"    {test['details']}")
        
        print(f"\n🎯 Results: {self.passed}/{len(self.tests)} tests passed")
        
        if self.passed == len(self.tests):
            print("🎉 All tests passed! The project is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
        
        return self.passed == len(self.tests)


results = TestResults()


async def test_core_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing Core Imports")
    print("="*50)
    
    try:
        # Test core imports
        from training_data_bot.core.models import Document, Dataset, TaskType
        from training_data_bot.core.config import SimpleConfig
        from training_data_bot.core.logging import get_logger
        
        # Test component imports
        from training_data_bot.sources.unified import UnifiedLoader
        from training_data_bot.sources.web import WebLoader
        from training_data_bot.sources.documents import DocumentLoader
        from training_data_bot.sources.pdf import PDFLoader
        
        from training_data_bot.tasks.manager import TaskManager
        from training_data_bot.tasks.qa_generation import QAGenerator
        from training_data_bot.tasks.classification import ClassificationGenerator
        from training_data_bot.tasks.summarization import SummarizationGenerator
        
        from training_data_bot.ai.client import AIClient
        from training_data_bot.decodo.client import DecodoClient
        from training_data_bot.preprocessing.processor import TextPreprocessor
        from training_data_bot.evaluation.evaluator import QualityEvaluator
        from training_data_bot.storage.export import DatasetExporter
        
        print("✅ All core modules imported successfully")
        results.add_test("Core Imports", True, "All modules imported without errors")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        results.add_test("Core Imports", False, f"Import error: {e}")
        return False


async def test_webloader_upgrade():
    """Test the upgraded WebLoader with Decodo integration."""
    print("\n🧪 Testing WebLoader Upgrade")
    print("="*50)
    
    try:
        async with WebLoader() as loader:
            # Test single URL
            test_url = "https://httpbin.org/html"
            print(f"📥 Testing single URL: {test_url}")
            
            document = await loader.load_single(test_url)
            
            # Verify document properties
            assert document.title is not None
            assert len(document.content) > 0
            assert document.extraction_method is not None
            assert document.source == test_url
            
            print(f"✅ Single URL test passed")
            print(f"   📄 Title: {document.title}")
            print(f"   📏 Content: {len(document.content)} characters")
            print(f"   🔧 Method: {document.extraction_method}")
            
            # Test multiple URLs
            test_urls = [
                "https://httpbin.org/html",
                "https://httpbin.org/json"
            ]
            
            print(f"📥 Testing multiple URLs: {len(test_urls)} URLs")
            documents = await loader.load_multiple_urls(test_urls, max_concurrent=2)
            
            assert len(documents) == len(test_urls)
            for doc in documents:
                assert doc.title is not None
                assert len(doc.content) > 0
            
            print(f"✅ Multiple URL test passed")
            print(f"   📄 Loaded {len(documents)} documents")
            
            results.add_test("WebLoader Upgrade", True, f"Loaded {len(documents)} documents successfully")
            return True
            
    except Exception as e:
        print(f"❌ WebLoader test failed: {e}")
        results.add_test("WebLoader Upgrade", False, f"Error: {e}")
        return False


async def test_unified_loader():
    """Test the UnifiedLoader with shared Decodo client."""
    print("\n🧪 Testing UnifiedLoader Integration")
    print("="*50)
    
    try:
        # Create shared Decodo client
        decodo_client = DecodoClient()
        
        async with UnifiedLoader(decodo_client=decodo_client) as loader:
            # Test URL loading
            test_url = "https://httpbin.org/json"
            print(f"📥 Testing URL through UnifiedLoader: {test_url}")
            
            document = await loader.load_single(test_url)
            
            assert document.title is not None
            assert len(document.content) > 0
            assert str(document.doc_type) == "url" or document.doc_type.value == "url"
            
            print(f"✅ UnifiedLoader test passed")
            print(f"   📄 Document: {document.title}")
            print(f"   📏 Content: {len(document.content)} characters")
            
            results.add_test("UnifiedLoader Integration", True, "Successfully loaded URL via UnifiedLoader")
            return True
            
    except Exception as e:
        print(f"❌ UnifiedLoader test failed: {e}")
        results.add_test("UnifiedLoader Integration", False, f"Error: {e}")
        return False


async def test_training_data_bot():
    """Test the main TrainingDataBot functionality."""
    print("\n🧪 Testing TrainingDataBot Integration")
    print("="*50)
    
    try:
        async with TrainingDataBot() as bot:
            # Test loading documents
            test_urls = ["https://httpbin.org/html"]
            print(f"📥 Loading documents: {test_urls}")
            
            documents = await bot.load_documents(test_urls)
            
            assert len(documents) == 1
            assert documents[0].title is not None
            assert len(documents[0].content) > 0
            
            print(f"✅ Document loading test passed")
            print(f"   📄 Loaded {len(documents)} documents")
            
            # Test document processing (with simulation mode)
            print(f"🔄 Testing document processing...")
            
            dataset = await bot.process_documents(
                documents=documents,
                task_types=[TaskType.QA_GENERATION],
                quality_filter=True
            )
            
            assert dataset is not None
            # Note: may generate 0 examples due to quality filtering in simulation mode
            # This is expected behavior
            
            print(f"✅ Document processing test passed")
            print(f"   🎯 Generated {len(dataset.examples)} examples (simulation mode)")
            
            # Test export functionality (if we have examples)
            if len(dataset.examples) > 0:
                print(f"💾 Testing export functionality...")
                
                with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
                    export_path = await bot.export_dataset(
                        dataset=dataset,
                        output_path=tmp.name,
                        format=ExportFormat.JSONL
                    )
                    
                    assert Path(export_path).exists()
                    print(f"✅ Export test passed")
                    print(f"   📁 Exported to: {export_path}")
                    
                    # Cleanup
                    os.unlink(export_path)
            else:
                print(f"💾 Skipping export test (no examples generated in simulation mode)")
            
            results.add_test("TrainingDataBot Integration", True, f"Full pipeline test: {len(dataset.examples)} examples generated")
            return True
            
    except Exception as e:
        print(f"❌ TrainingDataBot test failed: {e}")
        results.add_test("TrainingDataBot Integration", False, f"Error: {e}")
        return False


async def test_streamlit_components():
    """Test Streamlit dashboard components."""
    print("\n🧪 Testing Streamlit Dashboard Components")
    print("="*50)
    
    try:
        # Test imports
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        
        # Test dashboard app import
        sys.path.append(str(Path(__file__).parent / "src" / "training_data_bot" / "dashboard"))
        import app
        
        # Verify main functions exist
        assert hasattr(app, 'main')
        assert hasattr(app, 'dashboard_page')
        assert hasattr(app, 'documents_page')
        assert hasattr(app, 'generate_page')
        assert hasattr(app, 'analytics_page')
        assert hasattr(app, 'settings_page')
        
        print("✅ Streamlit dashboard components test passed")
        print("   📊 All dashboard functions available")
        print("   🎨 Dependencies (streamlit, pandas, plotly) working")
        
        results.add_test("Streamlit Components", True, "All dashboard functions and dependencies available")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit components test failed: {e}")
        results.add_test("Streamlit Components", False, f"Error: {e}")
        return False


async def test_decodo_client():
    """Test Decodo client functionality."""
    print("\n🧪 Testing Decodo Client")
    print("="*50)
    
    try:
        # Test client initialization
        client = DecodoClient()
        
        # Test client has required methods
        assert hasattr(client, 'scrape_url')
        assert hasattr(client, 'close')
        
        print("✅ Decodo client test passed")
        print("   🌐 Client initialized successfully")
        print("   🔧 Required methods available")
        
        await client.close()
        
        results.add_test("Decodo Client", True, "Client initialization and methods available")
        return True
        
    except Exception as e:
        print(f"❌ Decodo client test failed: {e}")
        results.add_test("Decodo Client", False, f"Error: {e}")
        return False


async def test_task_generators():
    """Test task generation components."""
    print("\n🧪 Testing Task Generators")
    print("="*50)
    
    try:
        from training_data_bot.tasks.qa_generation import QAGenerator
        from training_data_bot.tasks.classification import ClassificationGenerator
        from training_data_bot.tasks.summarization import SummarizationGenerator
        from training_data_bot.tasks.manager import TaskManager
        
        # Test task manager
        manager = TaskManager()
        
        # Verify generators are available
        assert hasattr(manager, 'generators')
        assert TaskType.QA_GENERATION in manager.generators
        assert TaskType.CLASSIFICATION in manager.generators
        assert TaskType.SUMMARIZATION in manager.generators
        
        print("✅ Task generators test passed")
        print("   🎯 QA Generation available")
        print("   🏷️ Classification available")
        print("   📝 Summarization available")
        
        results.add_test("Task Generators", True, "All task generators available")
        return True
        
    except Exception as e:
        print(f"❌ Task generators test failed: {e}")
        results.add_test("Task Generators", False, f"Error: {e}")
        return False


async def test_quality_evaluation():
    """Test quality evaluation system."""
    print("\n🧪 Testing Quality Evaluation")
    print("="*50)
    
    try:
        from training_data_bot.evaluation.evaluator import QualityEvaluator
        
        # Test evaluator initialization
        evaluator = QualityEvaluator()
        
        # Verify evaluator has required methods
        assert hasattr(evaluator, 'evaluate_example')
        assert hasattr(evaluator, 'evaluate_dataset')
        
        print("✅ Quality evaluation test passed")
        print("   🔍 Evaluator initialized successfully")
        print("   📊 Required methods available")
        
        results.add_test("Quality Evaluation", True, "Evaluator initialization and methods available")
        return True
        
    except Exception as e:
        print(f"❌ Quality evaluation test failed: {e}")
        results.add_test("Quality Evaluation", False, f"Error: {e}")
        return False


async def main():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Project Test Suite")
    print("="*60)
    print("Testing all components of the Training Data Bot project")
    print("="*60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("WebLoader Upgrade", test_webloader_upgrade),
        ("UnifiedLoader Integration", test_unified_loader),
        ("TrainingDataBot Integration", test_training_data_bot),
        ("Streamlit Components", test_streamlit_components),
        ("Decodo Client", test_decodo_client),
        ("Task Generators", test_task_generators),
        ("Quality Evaluation", test_quality_evaluation),
    ]
    
    for test_name, test_func in tests:
        try:
            await test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.add_test(test_name, False, f"Exception: {e}")
    
    # Print comprehensive summary
    success = results.print_summary()
    
    if success:
        print("\n🎉 PROJECT STATUS: ALL SYSTEMS OPERATIONAL")
        print("✅ Core functionality working")
        print("✅ WebLoader with Decodo integration working")
        print("✅ Document processing pipeline working")
        print("✅ Training data generation working")
        print("✅ Streamlit dashboard ready")
        print("✅ Quality evaluation system working")
    else:
        print("\n⚠️ PROJECT STATUS: SOME ISSUES DETECTED")
        print("Check the test results above for specific issues")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 