"""CSS styles and theme definitions for the Markit UI."""

# Main CSS styles for the application
CSS_STYLES = """
        /* Global styles */
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Document converter styles */
        .output-container {
            max-height: 420px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
        }
        
        .gradio-container .prose {
            overflow: visible;
        }
        
        .processing-controls { 
            display: flex; 
            justify-content: center; 
            gap: 10px; 
            margin-top: 10px; 
        }
        
        .provider-options-row {
            margin-top: 15px;
            margin-bottom: 15px;
        }
        
        /* Chat Tab Styles - Complete redesign */
        .chat-tab-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .chat-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .chat-header h2 {
            margin: 0;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        .chat-header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        /* Status Card Styling */
        .status-card {
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .status-card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        .status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f2f5;
        }
        
        .status-header h3 {
            margin: 0;
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .status-indicator {
            padding: 8px 16px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9em;
            letter-spacing: 0.5px;
        }
        
        .status-ready {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-not-ready {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .status-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        
        .status-label {
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .status-value {
            font-size: 1.4em;
            font-weight: 700;
            color: #495057;
        }
        
        .status-services {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .service-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 15px;
            border-radius: 8px;
            font-weight: 500;
            flex: 1;
            min-width: 200px;
            color: #2c3e50 !important;
        }
        
        .service-status span {
            color: #2c3e50 !important;
        }
        
        .service-ready {
            background: #d4edda;
            color: #2c3e50 !important;
            border: 1px solid #c3e6cb;
        }
        
        .service-ready span {
            color: #2c3e50 !important;
        }
        
        .service-error {
            background: #f8d7da;
            color: #2c3e50 !important;
            border: 1px solid #f5c6cb;
        }
        
        .service-error span {
            color: #2c3e50 !important;
        }
        
        .service-icon {
            font-size: 1.2em;
        }
        
        .service-indicator {
            margin-left: auto;
        }
        
        .status-error {
            border-color: #dc3545;
            background: #f8d7da;
        }
        
        .error-message {
            color: #721c24;
            margin: 0;
            font-weight: 500;
        }
        
        /* Control buttons styling */
        .control-buttons {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            margin-bottom: 25px;
        }
        
        .control-btn {
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
        }
        
        .btn-refresh {
            background: #17a2b8;
            color: white;
        }
        
        .btn-refresh:hover {
            background: #138496;
            transform: translateY(-1px);
        }
        
        .btn-new-session {
            background: #28a745;
            color: white;
        }
        
        .btn-new-session:hover {
            background: #218838;
            transform: translateY(-1px);
        }
        
        .btn-clear-data {
            background: #dc3545;
            color: white;
        }
        
        .btn-clear-data:hover {
            background: #c82333;
            transform: translateY(-1px);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Chat interface styling */
        .chat-main-container {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            overflow: hidden;
            margin-bottom: 25px;
        }
        
        .chat-container {
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid #e1e5e9;
            overflow: hidden;
        }
        
        /* Custom chatbot styling */
        .gradio-chatbot {
            border: none !important;
            background: #ffffff;
        }
        
        .gradio-chatbot .message {
            padding: 15px 20px;
            margin: 10px;
            border-radius: 12px;
        }
        
        .gradio-chatbot .message.user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 50px;
        }
        
        .gradio-chatbot .message.assistant {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            margin-right: 50px;
        }
        
        /* Input area styling */
        .chat-input-container {
            background: #ffffff;
            padding: 20px;
            border-top: 1px solid #e1e5e9;
            border-radius: 0 0 15px 15px;
        }
        
        .input-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
            resize: none;
            max-height: 120px;
            min-height: 48px;
        }
        
        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }
        
        .send-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            min-width: 80px;
            height: 48px;
            margin-right: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1em;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .send-button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Session info styling */
        .session-info {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            color: #0056b3;
            font-weight: 500;
            text-align: center;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .chat-tab-container {
                padding: 10px;
            }
            
            .status-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .service-status {
                min-width: 100%;
            }
            
            .control-buttons {
                flex-direction: column;
                gap: 8px;
            }
            
            .gradio-chatbot .message.user {
                margin-left: 20px;
            }
            
            .gradio-chatbot .message.assistant {
                margin-right: 20px;
            }
        }
        
        /* Query Ranker Styles */
        .ranker-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .ranker-placeholder {
            text-align: center;
            padding: 40px;
            background: #f8f9fa;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            color: #6c757d;
        }
        
        .ranker-placeholder h3 {
            color: #495057;
            margin-bottom: 10px;
        }
        
        .ranker-error {
            text-align: center;
            padding: 30px;
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 12px;
            color: #721c24;
        }
        
        .ranker-error h3 {
            margin-bottom: 15px;
        }
        
        .error-hint {
            font-style: italic;
            margin-top: 10px;
            opacity: 0.8;
        }
        
        .ranker-no-results {
            text-align: center;
            padding: 40px;
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            color: #6c757d;
        }
        
        .ranker-no-results h3 {
            color: #495057;
            margin-bottom: 15px;
        }
        
        .no-results-hint {
            font-style: italic;
            margin-top: 10px;
            opacity: 0.8;
        }
        
        .ranker-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .ranker-title h3 {
            margin: 0 0 10px 0;
            font-size: 1.4em;
            font-weight: 600;
        }
        
        .query-display {
            font-size: 1.1em;
            opacity: 0.9;
            font-style: italic;
            margin-bottom: 15px;
        }
        
        .ranker-meta {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .method-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 0.9em;
        }
        
        .result-count {
            background: rgba(255, 255, 255, 0.15);
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 0.9em;
        }
        
        .result-card {
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            overflow: hidden;
        }
        
        .result-card:hover {
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }
        
        .rank-info {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .rank-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .source-info {
            background: #e9ecef;
            color: #495057;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .page-info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }
        
        .length-info {
            background: #f8f9fa;
            color: #6c757d;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.85em;
        }
        
        .score-info {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .confidence-badge {
            padding: 4px 8px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.85em;
        }
        
        .score-value {
            background: #2c3e50;
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .result-content {
            padding: 20px;
        }
        
        .content-text {
            line-height: 1.6;
            color: #2c3e50;
            border-left: 3px solid #667eea;
            padding-left: 15px;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .result-actions {
            display: flex;
            gap: 10px;
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
        }
        
        .action-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .copy-btn {
            background: #17a2b8;
            color: white;
        }
        
        .copy-btn:hover {
            background: #138496;
            transform: translateY(-1px);
        }
        
        .info-btn {
            background: #6c757d;
            color: white;
        }
        
        .info-btn:hover {
            background: #5a6268;
            transform: translateY(-1px);
        }
        
        .ranker-methods {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #e9ecef;
        }
        
        .methods-label {
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .methods-list {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        .method-tag {
            background: #e9ecef;
            color: #495057;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }
        
        /* Ranker controls styling */
        .ranker-controls {
            background: #ffffff;
            border: 1px solid #e1e5e9;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .ranker-input-row {
            display: flex;
            gap: 15px;
            align-items: end;
            margin-bottom: 15px;
        }
        
        .ranker-query-input {
            flex: 1;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .ranker-query-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }
        
        .ranker-search-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            min-width: 100px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 1em;
        }
        
        .ranker-search-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .ranker-options-row {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        /* Responsive design for ranker */
        @media (max-width: 768px) {
            .ranker-container {
                padding: 10px;
            }
            
            .ranker-input-row {
                flex-direction: column;
                gap: 10px;
            }
            
            .ranker-options-row {
                flex-direction: column;
                gap: 10px;
                align-items: stretch;
            }
            
            .ranker-meta {
                justify-content: center;
            }
            
            .rank-info {
                flex-direction: column;
                gap: 5px;
                align-items: flex-start;
            }
            
            .result-header {
                flex-direction: column;
                gap: 10px;
                align-items: flex-start;
            }
            
            .score-info {
                align-self: flex-end;
            }
            
            .result-actions {
                flex-direction: column;
                gap: 8px;
            }
        }
"""