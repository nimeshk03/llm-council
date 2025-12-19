#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
import numpy as np
from enum import Enum
import re

class ExpertType(Enum):
    MATH = "math"
    CODING = "coding"
    VISION = "vision"
    KNOWLEDGE = "knowledge"
    RESEARCH = "research"

class SemanticRouter:
    def __init__(self):
        print("Loading semantic router (weighted keywords)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Weighted keywords: (pattern, weight)
        # Higher weight = stronger signal for that expert
        self.weighted_keywords = {
            ExpertType.MATH: [
                (r'\b(integral|integrate|derivative|differentiate|antiderivative)\b', 10.0),
                (r'\b(∫|∂|d/dx|dy/dx)\b', 10.0),
                (r'\b(limit|series|theorem|proof)\b', 8.0),
                (r'\b(probability|variance|deviation|expected)\b', 6.0),
                (r'\b(sin|cos|tan|log|ln|exp)\b', 5.0),
            ],
            ExpertType.CODING: [
                (r'\b(implement|write|code|create|build|debug|fix)\b', 6.0),
                (r'\b(function|algorithm|class|program|method|api)\b', 5.0),
                (r'\b(c\+\+|python|java|javascript|typescript|rust|go|php|c#|ruby)\b', 9.0),
                (r'\b(DFS|BFS|search|sort|tree|graph|linked list|hash table)\b', 7.0),
                (r'\b(array|string|loop|recursion|pointer)\b', 4.0),
            ],
            ExpertType.RESEARCH: [
                (r'\b(latest|recent|current|today|breaking)\b', 7.0),
                (r'\bnews\b', 8.0),
                (r'\b(ticker|stock|market|price|trading)\b', 9.0),
                (r'\b(AAPL|MSFT|GOOGL|TSLA|AMZN|NVDA|META|NFLX|AMD|INTC)\b', 10.0),
                (r'\b(cryptocurrency|bitcoin|ethereum|crypto)\b', 8.0),
                (r'\b(update|announcement|release)\b', 6.0),
            ],
            ExpertType.KNOWLEDGE: [
                (r'\b(explain|describe|teach|what is|how does|define)\b', 8.0),
                (r'\b(concept|principle|theory|definition)\b', 5.0),
            ],
        }
        
        # Fallback semantic examples (for queries without strong keywords)
        self.route_examples = {
            ExpertType.MATH: [
                "solve integral x squared",
                "find derivative of sine x",
                "calculate limit",
                "prove mathematical theorem"
            ],
            ExpertType.CODING: [
                "implement DFS in cpp",
                "write python function",
                "code binary search",
                "create API endpoint"
            ],
            ExpertType.RESEARCH: [
                "latest AAPL news",
                "current Tesla stock price",
                "recent AI updates",
                "today market trends"
            ],
            ExpertType.KNOWLEDGE: [
                "explain TCP/IP protocol",
                "what is object oriented programming",
                "describe REST architecture",
                "teach me binary trees"
            ],
            ExpertType.VISION: [
                "analyze this image",
                "read text from screenshot",
                "describe this picture"
            ]
        }
        
        print("Pre-computing fallback embeddings...")
        self.route_embeddings = {}
        for expert, examples in self.route_examples.items():
            embeddings = self.model.encode(
                examples, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
            self.route_embeddings[expert] = embeddings
        
        print("✓ SemanticRouter initialized")
    
    def route(self, query: str, has_image: bool = False, semantic_threshold: float = 0.4) -> ExpertType:
        """
        Route query to the appropriate expert using weighted keywords with semantic fallback.
        
        Args:
            query: User query string
            has_image: Whether query includes an image
            semantic_threshold: Minimum similarity score for semantic routing
            
        Returns:
            ExpertType indicating which expert should handle the query
        """
        if has_image:
            return ExpertType.VISION
        
        query_lower = query.lower().strip()
        
        # Step 1: Calculate weighted keyword scores
        # Use list to avoid any dict caching issues
        keyword_results = []
        
        for expert_type in [ExpertType.MATH, ExpertType.CODING, ExpertType.RESEARCH, ExpertType.KNOWLEDGE]:
            total_score = 0.0
            
            if expert_type in self.weighted_keywords:
                for pattern, weight in self.weighted_keywords[expert_type]:
                    if re.search(pattern, query_lower):
                        total_score += weight
            
            keyword_results.append((expert_type, total_score))
        
        # Find expert with highest keyword score
        best_keyword_expert, best_keyword_score = max(keyword_results, key=lambda x: x[1])
        
        # Debug output
        non_zero_scores = [(e, s) for e, s in keyword_results if s > 0]
        if non_zero_scores:
            score_str = ', '.join([f'{e.value}: {s:.1f}' for e, s in non_zero_scores])
            print(f"[DEBUG] Keyword scores: {{{score_str}}}")
            print(f"[DEBUG] → {best_keyword_expert.value} ({best_keyword_score:.1f})")
        
        if best_keyword_score > 0:
            return best_keyword_expert
        
        # Step 2: Fallback to semantic similarity when no keywords match
        print("[DEBUG] No keyword match, using semantic similarity...")
        
        query_embedding = self.model.encode(
            query, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        semantic_results = []
        for expert_type, example_embeddings in self.route_embeddings.items():
            # Compute cosine similarity with all examples
            similarities = np.dot(example_embeddings, query_embedding)
            max_similarity = float(np.max(similarities))
            semantic_results.append((expert_type, max_similarity))
        
        # Find expert with highest semantic similarity
        best_semantic_expert, best_semantic_score = max(semantic_results, key=lambda x: x[1])
        
        score_str = ', '.join([f'{e.value}: {s:.3f}' for e, s in semantic_results])
        print(f"[DEBUG] Semantic scores: {{{score_str}}}")
        print(f"[DEBUG] → {best_semantic_expert.value} ({best_semantic_score:.3f})")
        
        # Return best semantic match if above threshold, otherwise default to knowledge
        if best_semantic_score >= semantic_threshold:
            return best_semantic_expert
        else:
            print(f"[DEBUG] Score below threshold ({semantic_threshold}), using KNOWLEDGE")
            return ExpertType.KNOWLEDGE
