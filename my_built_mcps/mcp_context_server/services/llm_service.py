"""
services/llm_service.py

LLM service for text generation using Claude API.
"""
import asyncio
import logging
from typing import List, Dict, Optional
from anthropic import AsyncAnthropic

from ..config.settings import settings
from ..interfaces import AbstractSummarizer
from ..core.exceptions import LLMError

logger = logging.getLogger(__name__)


class ClaudeSummarizer(AbstractSummarizer):
    """Claude-based text summarization and topic extraction service."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.model = model or settings.LLM_MODEL
        self._client = None
    
    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client
    
    async def _call_claude(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.3,
        retries: int = 2
    ) -> Optional[str]:
        """Call Claude API with retry logic."""
        for attempt in range(retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Claude API attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    raise LLMError(f"Claude API failed after {retries} attempts: {e}")
        return None
    
    async def summarize(self, content: str, max_length: int = 500) -> str:
        """Generate intelligent summary using Claude."""
        try:
            text = await self._call_claude([{
                "role": "user",
                "content": f"""Summarize this conversation chunk concisely while preserving:
                1. Key technical decisions and implementations
                2. Important questions asked
                3. Code or architecture discussed
                4. Any tags or topics mentioned
                
                Conversation:
                {content[:4000]}
                
                Provide a 2-3 sentence summary."""
            }], max_tokens=max_length)
            
            return text if text else self._fallback_summary(content)
        except Exception as e:
            logger.error(f"LLM summary error: {e}")
            return self._fallback_summary(content)
    
    async def extract_topics(self, content: str, max_topics: int = 5) -> List[str]:
        """Extract topic anchors using LLM."""
        try:
            text = await self._call_claude([{
                "role": "user",
                "content": f"""Extract 3-5 key topic anchors from this conversation.
                Return only lowercase topic names separated by commas, like: api_design, error_handling, performance
                
                Conversation:
                {content[:2000]}
                
                Topics:"""
            }], max_tokens=100, temperature=0.2)
            
            if not text:
                return self._fallback_topic_extraction(content)
            
            topics_str = text.strip()
            # Validate Claude response
            if not topics_str or ',' not in topics_str:
                return []
            return [f"topic_{topic.strip()}" for topic in topics_str.split(',') if topic.strip()][:max_topics]
        except Exception as e:
            logger.error(f"LLM topic extraction error: {e}")
            return self._fallback_topic_extraction(content)
    
    def _fallback_summary(self, content: str) -> str:
        """Fallback summary generation."""
        lines = content.split('\n')
        return f"Summary of {len(lines)} messages discussing: {content[:200]}..."
    
    def _fallback_topic_extraction(self, content: str) -> List[str]:
        """Fallback topic extraction."""
        import re
        topics = set()
        tag_pattern = r'#(\w+)'
        topics.update(f"topic_{tag}" for tag in re.findall(tag_pattern, content.lower()))
        return list(topics)[:5]


# Default summarizer instance
summarizer = ClaudeSummarizer()