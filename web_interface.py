#!/usr/bin/env python3
import gradio as gr
from experts.session_manager import SessionManager
import socket

# Initialize once at startup
print("Initializing LLM Council system...")
mgr = SessionManager()
user_sessions = {}

def get_local_ip():
    """Get local IP address for network access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def respond(message, history):
    """Response function for ChatInterface."""
    # Use one global session
    if 'default' not in user_sessions:
        user_sessions['default'] = mgr.create_session()
    
    try:
        response = mgr.chat(user_sessions['default'], message)
        return response
    except Exception as e:
        return f"Error: {str(e)}\n\nTry rephrasing your question."

# Create interface with header
with gr.Blocks(title="LLM Council") as demo:
    gr.Markdown(
        """
        # LLM Council - Multi-Expert AI System
        
        **4 Specialized Experts:** Math | Coding | Research | Knowledge 
        
        Try: "Integral of x^3" • "Binary search in C++" • "NVDA news" • "Hypothesis testing"
        """
    )
    
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me anything...", container=False),
        examples=[
            "Find the integral of x^3",
            "Implement binary search in C++",
            "Latest NVIDIA news",
            "Explain hypothesis testing",
        ],
    )
    
    gr.Markdown("---\n**Powered by Ollama** • Weighted keyword routing • Context-aware")

if __name__ == "__main__":
    local_ip = get_local_ip()
    
    print("\n" + "="*70)
    print("LLM Council Web Interface")
    print("="*70)
    print(f"\nAccess URLs:")
    print(f"   • This laptop:  http://localhost:7860")
    print(f"   • Other device: http://{local_ip}:7860")
    print(f"\nMake sure both devices are on the same WiFi!")
    print(f"   If blocked: sudo ufw allow 7860")
    print("\nStarting server...\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
