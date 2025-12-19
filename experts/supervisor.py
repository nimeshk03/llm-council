#!/usr/bin/env python3

import requests
import time
from enum import Enum
from .rag_knowledge import KnowledgeRAG
from .research_expert import ResearchExpert
from .semantic_router import SemanticRouter, ExpertType

OLLAMA_URL = "http://localhost:11434"

EXPERT_MODELS = {
    ExpertType.MATH: "qwen2-math:7b-instruct",
    ExpertType.CODING: "deepseek-coder:6.7b-instruct",
    ExpertType.VISION: "llava:7b-v1.6",
    ExpertType.KNOWLEDGE: "qwen3:8b",
}


class Supervisor:
    def __init__(self):
        self.current_expert = None
        self.rag = KnowledgeRAG()
        self.research_expert = ResearchExpert()
        self.router = SemanticRouter()
        print("✓ Supervisor initialized")
        print("Experts:")
        for k, v in EXPERT_MODELS.items():
            print(f"  - {k.value}: {v}")

    def classify_query(self, query: str, has_image: bool = False) -> ExpertType:
        """Route query to appropriate expert."""
        return self.router.route(query, has_image=has_image)

    def load_expert(self, expert: ExpertType):
        """Load the specified expert model."""
        model = EXPERT_MODELS.get(expert)
        if not model:
            model = EXPERT_MODELS[ExpertType.KNOWLEDGE]
        
        if self.current_expert == expert:
            return
        
        # Unload previous expert
        if self.current_expert:
            prev_model = EXPERT_MODELS[self.current_expert]
            try:
                requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": prev_model, "prompt": "", "keep_alive": 0},
                    timeout=30
                )
            except:
                pass
        
        # Load new expert
        start = time.time()
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False},
            timeout=300
        )
        self.current_expert = expert
        print(f"Loaded {expert.value} expert ({model}) in {time.time() - start:.1f}s")

    def ask(self, query: str, image_path: str | None = None) -> str:
        """
        Process query with appropriate expert.
        OPTIMIZED: Reduced context size and timeout for knowledge queries.
        """
        has_image = image_path is not None
        
        # ========== DEBUG: IMAGE PATH ==========
        if image_path:
            print(f"[DEBUG] Received image_path: {image_path}")
        else:
            print(f"[DEBUG] No image_path provided")
        # =======================================
        
        expert = self.classify_query(query, has_image=has_image)
        print(f"Routing to: {expert.value}")
        
        # Research expert handles its own flow
        if expert == ExpertType.RESEARCH:
            return self.research_expert.answer_with_citations(query)
        
        # Load appropriate expert model
        self.load_expert(expert)
        model = EXPERT_MODELS[expert]
        final_prompt = query
        timeout = 120  # Default timeout
        
        # Knowledge expert: Use RAG with REDUCED context
        if expert == ExpertType.KNOWLEDGE:
            # Fetch only 2 chunks, max 500 chars each
            ctx = self.rag.build_context(
                query,
                k=2,  # Reduced from 4
                subject=None,
                source_filter=None,
                max_chars_per_chunk=500  # Truncate chunks
            )
            print("\n[DEBUG] RAG context used:\n")
            print(ctx[:600], "\n[...trimmed...]\n")
            
            # Simplified prompt
            final_prompt = f"""Using the context below, answer the question concisely.

Context:
{ctx}

Question: {query}

Answer (be brief, 2-3 sentences):"""
            timeout = 90  # Reduced from 300
        
        # Build payload
        payload = {
            "model": model,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "num_predict": 300 if expert == ExpertType.KNOWLEDGE else 512,
                "temperature": 0.3 if expert == ExpertType.KNOWLEDGE else 0.7,
            }
        }
        
        # ========== VISION EXPERT: ADD IMAGE WITH DEBUG ==========
        if expert == ExpertType.VISION:
            print(f"[DEBUG VISION] Expert is VISION")
            print(f"[DEBUG VISION] image_path provided: {image_path is not None}")
            
            if image_path:
                import base64
                import pathlib
                
                p = pathlib.Path(image_path)
                print(f"[DEBUG VISION] Path object: {p}")
                print(f"[DEBUG VISION] File exists: {p.exists()}")
                
                if p.exists():
                    file_size = p.stat().st_size
                    print(f"[DEBUG VISION] File size: {file_size / 1024:.1f} KB")
                    
                    try:
                        with p.open("rb") as f:
                            img_bytes = f.read()
                            b64_img = base64.b64encode(img_bytes).decode("utf-8")
                            payload["images"] = [b64_img]
                            print(f"[DEBUG VISION] Base64 encoded length: {len(b64_img)}")
                            print(f"[DEBUG VISION] Successfully added image to payload")
                    except Exception as e:
                        print(f"[DEBUG VISION] ERROR encoding image: {e}")
                else:
                    print(f"[DEBUG VISION] ERROR: File does not exist!")
            else:
                print(f"[DEBUG VISION] WARNING: Vision expert called but no image_path!")
            
            print(f"[DEBUG VISION] Payload has 'images' key: {'images' in payload}")
            if 'images' in payload:
                print(f"[DEBUG VISION] Number of images: {len(payload['images'])}")
        # =========================================================
        
        # Call LLM
        try:
            print(f"[DEBUG] Sending request to Ollama...")
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=timeout
            )
            print(f"[DEBUG] Response status: {r.status_code}")
            r.raise_for_status()
            response = r.json()["response"]
            print(f"[DEBUG] Response length: {len(response)} chars")
            return response
            
        except requests.exceptions.Timeout:
            return "Response timed out. The question may be too complex. Try asking a simpler version."
        except Exception as e:
            return f"Error: {str(e)}"
