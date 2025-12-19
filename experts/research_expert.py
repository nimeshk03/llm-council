#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List
import requests

OLLAMA_URL = "http://localhost:11434"
RESEARCH_MODEL = "qwen3:8b"

@dataclass
class WebSource:
    id: int
    title: str
    url: str
    snippet: str

class ResearchExpert:
    """Research Expert: Fetches real-time web information with citations."""
    
    def __init__(self):
        print("✓ ResearchExpert initialized (real web search)")
        try:
            from ddgs import DDGS
            self.ddgs_available = True
        except ImportError:
            print("[WARNING] ddgs not installed. Run: pip install ddgs")
            self.ddgs_available = False
    
    def search_web(self, query: str, max_results: int = 5) -> List[WebSource]:
        """Search the web using DuckDuckGo."""
        if not self.ddgs_available:
            return [
                WebSource(
                    id=1,
                    title="Install ddgs package for real results",
                    url="https://pypi.org/project/ddgs/",
                    snippet="Run: pip install ddgs",
                )
            ]
        
        try:
            from ddgs import DDGS
            sources = []
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)
            
            for idx, result in enumerate(results, start=1):
                sources.append(WebSource(
                    id=idx,
                    title=result.get('title', 'No title'),
                    url=result.get('href', ''),
                    snippet=result.get('body', 'No description')
                ))
                if idx >= max_results:
                    break
            
            return sources
        
        except Exception as e:
            print(f"[ERROR] Web search failed: {e}")
            return [
                WebSource(
                    id=1,
                    title="Search Error",
                    url="",
                    snippet=f"Could not complete search: {str(e)}"
                )
            ]
    
    def build_context(self, sources: List[WebSource]) -> str:
        """Format sources into context with [SOURCE n] labels."""
        if not sources:
            return "No sources available."
        
        parts = []
        for s in sources:
            parts.append(
                f"[SOURCE {s.id}]\n"
                f"Title: {s.title}\n"
                f"URL: {s.url}\n"
                f"Content: {s.snippet}\n"
            )
        
        return "\n---\n".join(parts)
    
    def answer_with_citations(self, question: str) -> str:
        """
        Full research pipeline:
        1. Search web
        2. Build context
        3. LLM synthesis with citations
        """
        # Step 1: Search
        print(f"[DEBUG] Searching web for: {question[:60]}...")
        sources = self.search_web(question, max_results=5)
        
        if not sources:
            return "I couldn't find any information on that topic."
        
        # Step 2: Build context
        context = self.build_context(sources)
        
        # ========== DEBUG CONTEXT ==========
        print(f"[DEBUG] Context length: {len(context)} chars")
        print(f"[DEBUG] Number of sources: {len(sources)}")
        print(f"[DEBUG] Context preview:\n{context[:400]}...\n")
        # ===================================
        
        # Step 3: Prompt
        prompt = f"""You are a research assistant. Answer using ONLY the sources provided below.

Sources:
{context}

Question: {question}

Instructions:
- Cite facts with [SOURCE n] tags (e.g., "Apple stock rose [SOURCE 1]")
- Be concise (2-3 sentences max)
- If sources lack info, say "The sources don't contain sufficient information"
- DO NOT make up information

Answer:"""
        
        # Step 4: LLM synthesis
        try:
            print("[DEBUG] Synthesizing answer with LLM...")
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": RESEARCH_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 300,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json()["response"].strip()
            
            # ========== DEBUG RESPONSE ==========
            print(f"[DEBUG] Raw LLM response: '{answer}'")
            print(f"[DEBUG] Response length: {len(answer)} chars")
            # ====================================
            
            if not answer or len(answer) < 10:
                print("[DEBUG] LLM returned empty/short response, using fallback")
                answer = "The sources don't contain sufficient information on this topic"
            
            # Append sources
            source_list = "\n\n**Sources:**"
            for s in sources:
                source_list += f"\n[{s.id}] {s.title}\n    {s.url}"
            
            return answer + source_list
        
        except requests.exceptions.Timeout:
            print("[ERROR] LLM synthesis timed out")
            # Return raw results as fallback
            fallback = "**Search Results:**\n\n"
            for s in sources:
                fallback += f"**[{s.id}] {s.title}**\n{s.snippet}\n{s.url}\n\n"
            return fallback
        
        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            fallback = "**Search Results:**\n\n"
            for s in sources:
                fallback += f"**[{s.id}] {s.title}**\n{s.snippet}\n{s.url}\n\n"
            return fallback
