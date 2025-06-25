#!/usr/bin/env python3
"""
Test script to verify the Phase 1 implementation structure is correct.
This test checks imports, method signatures, and class structure without requiring API keys.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all new imports work correctly."""
    print("🔧 Testing Imports and Structure")
    print("=" * 40)
    
    try:
        # Test vector store imports
        from src.rag.vector_store import VectorStoreManager, vector_store_manager
        print("✅ VectorStoreManager imports successfully")
        
        # Test chat service imports  
        from src.rag.chat_service import RAGChatService, rag_chat_service
        print("✅ RAGChatService imports successfully")
        
        # Test LangChain community imports
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        print("✅ BM25Retriever and EnsembleRetriever import successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_method_signatures():
    """Test that all new methods have correct signatures."""
    print("\n🔍 Testing Method Signatures")
    print("=" * 40)
    
    try:
        from src.rag.vector_store import VectorStoreManager
        from src.rag.chat_service import RAGChatService
        
        # Test VectorStoreManager methods
        vm = VectorStoreManager()
        
        # Check method exists
        assert hasattr(vm, 'get_bm25_retriever'), "get_bm25_retriever method missing"
        assert hasattr(vm, 'get_hybrid_retriever'), "get_hybrid_retriever method missing"
        print("✅ VectorStoreManager has new methods")
        
        # Test RAGChatService methods
        cs = RAGChatService()
        
        assert hasattr(cs, 'chat_with_retrieval'), "chat_with_retrieval method missing"
        assert hasattr(cs, 'chat_stream_with_retrieval'), "chat_stream_with_retrieval method missing"
        assert hasattr(cs, 'set_default_retrieval_method'), "set_default_retrieval_method method missing"
        print("✅ RAGChatService has new methods")
        
        # Test method parameters (basic signature check)
        import inspect
        
        # Check get_hybrid_retriever signature
        sig = inspect.signature(vm.get_hybrid_retriever)
        expected_params = ['k', 'semantic_weight', 'keyword_weight', 'search_type', 'search_kwargs']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"Parameter {param} missing from get_hybrid_retriever"
        print("✅ get_hybrid_retriever has correct parameters")
        
        # Check chat_with_retrieval signature
        sig = inspect.signature(cs.chat_with_retrieval)
        expected_params = ['user_message', 'retrieval_method', 'retrieval_config']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"Parameter {param} missing from chat_with_retrieval"
        print("✅ chat_with_retrieval has correct parameters")
        
        return True
    except Exception as e:
        print(f"❌ Method signature test failed: {e}")
        return False

def test_class_attributes():
    """Test that classes have the required new attributes."""
    print("\n📋 Testing Class Attributes")
    print("=" * 40)
    
    try:
        from src.rag.vector_store import VectorStoreManager
        from src.rag.chat_service import RAGChatService
        
        # Test VectorStoreManager attributes
        vm = VectorStoreManager()
        assert hasattr(vm, '_documents_cache'), "_documents_cache attribute missing"
        assert hasattr(vm, '_bm25_retriever'), "_bm25_retriever attribute missing"
        print("✅ VectorStoreManager has new attributes")
        
        # Test RAGChatService attributes
        cs = RAGChatService()
        assert hasattr(cs, '_current_retrieval_method'), "_current_retrieval_method attribute missing"
        assert hasattr(cs, '_default_retrieval_method'), "_default_retrieval_method attribute missing"
        assert hasattr(cs, '_default_retrieval_config'), "_default_retrieval_config attribute missing"
        print("✅ RAGChatService has new attributes")
        
        return True
    except Exception as e:
        print(f"❌ Class attributes test failed: {e}")
        return False

def test_configuration_options():
    """Test that different configuration options can be set."""
    print("\n⚙️ Testing Configuration Options")
    print("=" * 40)
    
    try:
        from src.rag.chat_service import rag_chat_service
        
        # Test setting different retrieval methods
        configs = [
            ("similarity", {"k": 4}),
            ("mmr", {"k": 3, "fetch_k": 10, "lambda_mult": 0.5}),
            ("hybrid", {"k": 4, "semantic_weight": 0.7, "keyword_weight": 0.3})
        ]
        
        for method, config in configs:
            try:
                rag_chat_service.set_default_retrieval_method(method, config)
                assert rag_chat_service._default_retrieval_method == method
                assert rag_chat_service._default_retrieval_config == config
                print(f"✅ {method} configuration works")
            except Exception as e:
                print(f"❌ {method} configuration failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_requirements_updated():
    """Test that requirements.txt has the new dependencies."""
    print("\n📦 Testing Requirements Update")
    print("=" * 40)
    
    try:
        requirements_path = Path(__file__).parent / "requirements.txt"
        
        if requirements_path.exists():
            with open(requirements_path, 'r') as f:
                content = f.read()
            
            required_packages = [
                "langchain-community",
                "rank-bm25"
            ]
            
            for package in required_packages:
                if package in content:
                    print(f"✅ {package} found in requirements.txt")
                else:
                    print(f"❌ {package} missing from requirements.txt")
                    return False
            
            return True
        else:
            print("❌ requirements.txt not found")
            return False
            
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Run all structure tests."""
    print("🚀 Phase 1 Implementation Structure Tests")
    print("Testing code structure without requiring API keys")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Method Signatures", test_method_signatures), 
        ("Class Attributes", test_class_attributes),
        ("Configuration Options", test_configuration_options),
        ("Requirements Update", test_requirements_updated)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Structure Test Summary")
    print("=" * 40)
    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 Phase 1 Implementation Structure is PERFECT!")
        print("✅ All imports work correctly")
        print("✅ All method signatures are correct")
        print("✅ All class attributes are present") 
        print("✅ Configuration system works")
        print("✅ Requirements are updated")
        print("\n💡 The implementation is ready for use once API keys are configured!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} structure issues found")
        return 1

if __name__ == "__main__":
    exit(main())