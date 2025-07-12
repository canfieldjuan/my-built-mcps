"""
utils/helpers.py

General utility functions.
"""
import hashlib
from datetime import datetime
from typing import List, Dict, Any


def generate_node_id(session_id: str, content: str) -> str:
    """Generate a unique node ID."""
    timestamp = datetime.now().isoformat()
    data = f"{session_id}:{content}{timestamp}"
    return hashlib.md5(data.encode()).hexdigest()[:12]


def extract_keywords_from_text(text: str, min_length: int = 4, max_keywords: int = 20) -> List[str]:
    """Extract keywords from text."""
    words = text.lower().split()
    keywords = [w for w in words if len(w) > min_length]
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    return unique_keywords[:max_keywords]


def format_context_part(
    node: Dict[str, Any],
    include_timestamp: bool = True
) -> str:
    """Format a context node for display."""
    parts = []
    
    # Add timestamp if requested
    if include_timestamp and 'timestamp' in node:
        timestamp = datetime.fromisoformat(node['timestamp'])
        parts.append(f"[{timestamp.strftime('%Y-%m-%d %H:%M')}]")
    
    # Add tier indicator for non-short-term
    tier = node.get('tier', '')
    if tier and tier != 'short_term':
        tier_labels = {
            'mid_term': 'Summary',
            'long_term': 'Historical',
            'meta': 'Meta'
        }
        label = tier_labels.get(tier, tier.title())
        parts.append(f"[{label}]")
    
    # Add content
    content = node.get('summary') or node.get('content', '')
    if not content:
        content = "<empty>"
    
    parts.append(content)
    
    return ' '.join(parts) if parts else ""


def calculate_token_usage(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate token usage by tier."""
    usage = {
        'short_term': 0,
        'mid_term': 0,
        'long_term': 0,
        'meta': 0,
        'total': 0
    }
    
    for node in nodes:
        tokens = node.get('tokens', 0)
        tier = node.get('tier', 'mid_term')
        
        if tier in usage:
            usage[tier] += tokens
        usage['total'] += tokens
    
    return usage