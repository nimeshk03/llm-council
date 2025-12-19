#!/usr/bin/env python3

import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict
from .supervisor import Supervisor

@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str
    expert_used: str = ""  # Track which expert answered
    timestamp: float = field(default_factory=time.time)

@dataclass
class Session:
    id: str
    history: List[Turn] = field(default_factory=list)
    last_expert: str = ""  # Track last expert used
    last_topic: str = ""  # Track last topic/domain

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.supervisor = Supervisor()
        print("✓ SessionManager initialized (with context memory)")
    
    def create_session(self) -> str:
        """Create a new conversation session."""
        sid = str(uuid.uuid4())[:8]
        self.sessions[sid] = Session(id=sid)
        print(f"Created session {sid}")
        return sid
    
    def add_turn(self, sid: str, role: str, content: str, expert_used: str = ""):
        """Add a turn to session history."""
        self.sessions[sid].history.append(
            Turn(role=role, content=content, expert_used=expert_used)
        )
    
    def get_recent_context(self, sid: str, max_turns: int = 4) -> str:
        """
        Get recent conversation turns for context.
        Returns a formatted string with recent Q&A.
        """
        session = self.sessions[sid]
        recent_turns = session.history[-max_turns:] if len(session.history) > 0 else []
        
        if not recent_turns:
            return ""
        
        context_lines = []
        for turn in recent_turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            # Truncate long responses for context
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            context_lines.append(f"{prefix}: {content}")
        
        return "\n".join(context_lines)
    
    def enhance_query_with_context(self, sid: str, user_query: str) -> str:
        """
        Enhance user query with conversation context for better routing and responses.
        Makes implicit references explicit.
        """
        session = self.sessions[sid]
        
        # Get all previous user queries
        # NOTE: Current query has ALREADY been added to history by chat() method
        all_user_turns = [t for t in session.history if t.role == "user"]
        
        # We need at least 2 user turns (current + previous)
        if len(all_user_turns) < 2:
            return user_query
        
        # Get the PREVIOUS query (second to last, since last is current)
        last_query = all_user_turns[-2].content
        query_lower = user_query.lower().strip()
        
        # DEBUG: Print what we're working with
        print(f"[DEBUG] Previous query: '{last_query}'")
        print(f"[DEBUG] Current query: '{user_query}'")
        
        # Detect follow-up indicators
        # Detect follow-up indicators
        follow_up_indicators = [
            "now find", "now do", "now implement", "now write", "now calculate",
            "now evaluate", "now compute", "now solve",  # ← ADD THESE
            "also", "what about", "how about", "and the",
            "then", "next", "after that", "do the same", "same in"
        ]

        
        is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
        is_short = len(user_query.split()) <= 5
        
        print(f"[DEBUG] is_follow_up={is_follow_up}, is_short={is_short}")
        
        # Only process if it looks like a follow-up
        if is_follow_up or is_short:
            import re
            
            # Extract from PREVIOUS query
            math_expr = re.search(r'(x\s*\^\s*\d+|x\d+|sin\s*\([^)]+\)|cos\s*\([^)]+\)|e\s*\^\s*x)', last_query, re.I)
            stock_ticker = re.search(r'\b([A-Z]{2,5})\b(?:\s+(?:stock|ticker))', last_query)
            code_subject = re.search(r'(binary search|depth first search|breadth first search|DFS|BFS|quicksort|merge sort|linked list)', last_query, re.I)
            
            print(f"[DEBUG] math_expr={math_expr.group(1) if math_expr else None}")
            print(f"[DEBUG] stock_ticker={stock_ticker.group(1) if stock_ticker else None}")
            print(f"[DEBUG] code_subject={code_subject.group(1) if code_subject else None}")
            
            # Check keywords in current query
            has_derivative = "derivative" in query_lower
            has_integral = "integral" in query_lower
            print(f"[DEBUG] has_derivative={has_derivative}, has_integral={has_integral}")
            
            # Math follow-up
            if math_expr and (has_derivative or has_integral) and is_follow_up:
                subject = math_expr.group(1).replace(' ', '')
                enhanced = f"{user_query.rstrip('?.')} of {subject}"
                print(f"[DEBUG] ✓ Math follow-up triggered: '{enhanced}'")
                return enhanced
            
            # Stock follow-up
            elif ("what about" in query_lower or "how about" in query_lower) and stock_ticker:
                new_ticker = re.search(r'\b([A-Z]{2,5})\b', user_query)
                if new_ticker:
                    enhanced = last_query.replace(stock_ticker.group(1), new_ticker.group(1))
                    print(f"[DEBUG] ✓ Stock follow-up triggered: '{enhanced}'")
                    return enhanced
            
            # Code follow-up
            elif code_subject and ("do" in query_lower or "implement" in query_lower or "write" in query_lower):
                algo = code_subject.group(1)
                lang_pattern = r'\b(python|java|javascript|c\+\+|rust|go|ruby|c#)\b'
                new_lang_match = re.search(lang_pattern, query_lower)
                if new_lang_match:
                    new_lang = new_lang_match.group(1)
                    enhanced = f"implement {algo} in {new_lang}"
                    print(f"[DEBUG] ✓ Code follow-up triggered: '{enhanced}'")
                    return enhanced
            
            # Generic fallback
            elif is_follow_up:
                enhanced = f"Context: {last_query}\n\nFollow-up question: {user_query}"
                print(f"[DEBUG] → Generic follow-up (no specific pattern matched)")
                return enhanced
        
        return user_query
    
    def extract_image_path(self, query: str) -> tuple[str, str | None]:
        """
        Extract image path from query if present.
        Returns (clean_query, image_path)
        """
        import re
        import pathlib
    
        # Match: /path/with spaces/file.ext followed by text
        match = re.match(r'^([/~].+?\.(png|jpg|jpeg|gif|bmp|webp))\s+(.*)$', query, re.IGNORECASE)
    
        if match:
            image_path = match.group(1)
            clean_query = match.group(3)
        
            print(f"[DEBUG SESSION] Extracted image path: '{image_path}'")
            print(f"[DEBUG SESSION] Clean query: '{clean_query}'")
        
            # Verify file exists
            p = pathlib.Path(image_path)
            if p.exists():
                return clean_query, image_path
            else:
                print(f"[DEBUG SESSION] WARNING: Path doesn't exist: {image_path}")
    
        return query, None


    def chat(self, sid: str, user_message: str) -> str:
        """
        Process user message with contextual awareness.
        """
        # Extract image path if present
        clean_message, image_path = self.extract_image_path(user_message)
    
        # Store original user message
        self.add_turn(sid, "user", user_message)
    
        # Enhance query with context for better routing/response
        enhanced_query = self.enhance_query_with_context(sid, clean_message)
    
        # Get answer from supervisor (routing happens here)
        answer = self.supervisor.ask(enhanced_query, image_path=image_path)  # ← Pass image_path!
    
        # Track which expert was used
        expert_used = self.supervisor.current_expert.value if self.supervisor.current_expert else "unknown"
    
        # Store assistant response with metadata
        self.add_turn(sid, "assistant", answer, expert_used=expert_used)
    
        return answer


if __name__ == "__main__":
    mgr = SessionManager()
    sid = mgr.create_session()
    print(f"Using session: {sid}")
    print("\n Tip: Try follow-up questions like:")
    print("   - 'What about X?' (after asking about something)")
    print("   - 'Now do it in Python' (after code in another language)")
    print("   - 'Now find the derivative' (after an integral)")
    print()
    
    while True:
        try:
            q = input("\nYou: ")
            if q.strip().lower() in {"exit", "quit"}:
                print("\nGoodbye!")
                break
            
            if not q.strip():
                continue
            
            resp = mgr.chat(sid, q)
            print("\nAssistant:\n", resp)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
