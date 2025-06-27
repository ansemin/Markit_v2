#!/usr/bin/env python3
"""
Simple test script for Gemini wrapper functionality
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_gemini_wrapper():
    """Test Gemini wrapper without API key"""
    print("Testing Gemini wrapper structure...")
    
    try:
        from src.core.gemini_client_wrapper import (
            GeminiClientWrapper, 
            GeminiChatCompletions, 
            GeminiResponse,
            HAS_GEMINI,
            create_gemini_client_for_markitdown
        )
        print("✅ All classes imported successfully")
        print(f"✅ HAS_GEMINI: {HAS_GEMINI}")
        
        # Test response structure
        test_response = GeminiResponse("Test image description")
        print(f"✅ Response choices: {len(test_response.choices)}")
        print(f"✅ Message content: {test_response.choices[0].message.content}")
        
        # Test client creation (should fail gracefully without API key)
        client = create_gemini_client_for_markitdown()
        print(f"✅ Client creation (no API key): {client is None}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_markitdown_availability():
    """Test MarkItDown availability"""
    print("\nTesting MarkItDown availability...")
    
    try:
        from markitdown import MarkItDown
        print("✅ MarkItDown imported successfully")
        
        # Test basic initialization
        md = MarkItDown()
        print("✅ MarkItDown initialized without LLM client")
        
    except Exception as e:
        print(f"❌ MarkItDown error: {e}")
        return False
    
    return True

def test_integration_structure():
    """Test the overall integration structure"""
    print("\nTesting integration structure...")
    
    try:
        # Test that our wrapper can theoretically work with MarkItDown
        from src.core.gemini_client_wrapper import GeminiClientWrapper, HAS_GEMINI
        from markitdown import MarkItDown
        
        print("✅ Both components available for integration")
        
        # Test interface compatibility (structure only)
        if HAS_GEMINI:
            print("✅ Gemini dependency available")
        else:
            print("⚠️  Gemini dependency not available")
            
        print("✅ Integration structure test passed")
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Testing Gemini-MarkItDown Integration ===\n")
    
    success = True
    success &= test_gemini_wrapper()
    success &= test_markitdown_availability() 
    success &= test_integration_structure()
    
    print(f"\n=== Overall Result: {'✅ PASS' if success else '❌ FAIL'} ===")