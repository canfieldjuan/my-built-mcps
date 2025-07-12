"""
policies/recall_policy.py

Context recall and retrieval policies.
"""
from typing import Dict, Any, List, Tuple
from datetime import datetime

from ..models.enums import MemoryTier


class RecallPolicy:
    """Defines policies for context recall and retrieval."""
    
    def __init__(
        self,
        relevance_threshold: float = 0.7,
        max_context_ratio: Dict[MemoryTier, float] = None
    ):
        self.relevance_threshold = relevance_threshold
        self.max_context_ratio = max_context_ratio or {
            MemoryTier.SHORT_TERM: 0.3,
            MemoryTier.MID_TERM: 0.5,
            MemoryTier.LONG_TERM: 0.2
        }
    
    def filter_by_relevance(
        self,
        nodes: List[Dict[str, Any]],
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Filter nodes by relevance score."""
        threshold = threshold or self.relevance_threshold
        return [
            node for node in nodes
            if node.get('similarity', 0) >= threshold
        ]
    
    def allocate_token_budget(
        self,
        max_tokens: int,
        has_short_term: bool = True,
        has_relevant: bool = True
    ) -> Dict[str, int]:
        """Allocate token budget across different sources."""
        budgets = {}
        
        # Reserve tokens for short-term buffer
        if has_short_term:
            budgets['short_term'] = int(max_tokens * self.max_context_ratio[MemoryTier.SHORT_TERM])
        
        # Reserve tokens for mid-term summaries
        budgets['mid_term'] = int(max_tokens * self.max_context_ratio[MemoryTier.MID_TERM])
        
        # Reserve tokens for relevant long-term context
        if has_relevant:
            budgets['long_term'] = int(max_tokens * self.max_context_ratio[MemoryTier.LONG_TERM])
        
        # Remaining tokens for overflow
        used = sum(budgets.values())
        budgets['overflow'] = max_tokens - used
        
        return budgets
    
    def rank_nodes(
        self,
        nodes: List[Dict[str, Any]],
        query_embedding: List[float] = None
    ) -> List[Dict[str, Any]]:
        """Rank nodes by multiple factors."""
        # Sort by multiple criteria
        def score_node(node: Dict[str, Any]) -> Tuple[float, float, float]:
            # Primary: Relevance score
            relevance = node.get('similarity', 0)
            
            # Secondary: Tier priority (short-term > mid-term > long-term)
            tier_priority = {
                MemoryTier.SHORT_TERM.value: 3,
                MemoryTier.MID_TERM.value: 2,
                MemoryTier.LONG_TERM.value: 1,
                MemoryTier.META.value: 0
            }
            tier_score = tier_priority.get(node.get('tier', ''), 0)
            
            # Tertiary: Recency
            timestamp = node.get('timestamp', '')
            if timestamp:
                node_time = datetime.fromisoformat(timestamp)
                age_hours = (datetime.now() - node_time).total_seconds() / 3600
                recency_score = 1 / (1 + age_hours / 24)  # Decay over days
            else:
                recency_score = 0
            
            return (relevance, tier_score, recency_score)
        
        return sorted(nodes, key=score_node, reverse=True)
    
    def should_include_node(
        self,
        node: Dict[str, Any],
        current_tokens: int,
        max_tokens: int,
        tier_budget: Dict[str, int],
        tier_usage: Dict[str, int]
    ) -> bool:
        """Determine if a node should be included in context."""
        node_tokens = node.get('tokens', 0)
        node_tier = node.get('tier', MemoryTier.MID_TERM.value)
        
        # Check total token limit
        if current_tokens + node_tokens > max_tokens:
            return False
        
        # Check tier-specific budget
        tier_key = {
            MemoryTier.SHORT_TERM.value: 'short_term',
            MemoryTier.MID_TERM.value: 'mid_term',
            MemoryTier.LONG_TERM.value: 'long_term',
            MemoryTier.META.value: 'long_term'
        }.get(node_tier, 'overflow')
        
        current_tier_usage = tier_usage.get(tier_key, 0)
        tier_limit = tier_budget.get(tier_key, 0)
        
        return current_tier_usage + node_tokens <= tier_limit


# Default recall policy
default_recall_policy = RecallPolicy()