"""
policies/tier_policy.py

Memory tier management policies.
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta

from ..models.enums import MemoryTier


class TierPolicy:
    """Defines policies for memory tier transitions."""
    
    def __init__(
        self,
        mid_term_retention_count: int = 10,
        long_term_age_hours: int = 24,
        meta_summary_threshold: int = 50
    ):
        self.mid_term_retention_count = mid_term_retention_count
        self.long_term_age_hours = long_term_age_hours
        self.meta_summary_threshold = meta_summary_threshold
    
    def should_promote_to_long_term(
        self,
        node: Dict[str, Any],
        total_mid_term_count: int
    ) -> bool:
        """Determine if a node should be promoted to long-term."""
        # Check if we have too many mid-term nodes
        if total_mid_term_count > self.mid_term_retention_count:
            return True
        
        # Check age
        if 'timestamp' in node:
            node_time = datetime.fromisoformat(node['timestamp'])
            age = datetime.now() - node_time
            if age > timedelta(hours=self.long_term_age_hours):
                return True
        
        return False
    
    def should_create_meta_summary(
        self,
        long_term_count: int
    ) -> bool:
        """Determine if we should create a meta-level summary."""
        return long_term_count >= self.meta_summary_threshold
    
    def get_nodes_to_promote(
        self,
        mid_term_nodes: List[Dict[str, Any]],
        keep_recent: int = None
    ) -> List[Dict[str, Any]]:
        """Get list of nodes that should be promoted."""
        keep_recent = keep_recent or self.mid_term_retention_count
        
        if len(mid_term_nodes) <= keep_recent:
            return []
        
        # Sort by timestamp (oldest first)
        sorted_nodes = sorted(
            mid_term_nodes,
            key=lambda x: x.get('timestamp', ''),
            reverse=False
        )
        
        # Return nodes to promote (oldest ones)
        return sorted_nodes[:-keep_recent]


# Default tier policy
default_tier_policy = TierPolicy()