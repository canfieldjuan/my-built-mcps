#!/usr/bin/env python3
"""
Shopify AI Assistant - Complete SaaS Architecture
Merchants install app → AI works immediately → Monthly subscription
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import aiosqlite
from fastmcp import FastMCP, Context
from pydantic import BaseModel
import shopify
import logging

logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====

# Your master Claude API keys (you own these)
CLAUDE_API_KEYS = [
    "sk-ant-your-key-1",  # Primary
    "sk-ant-your-key-2",  # Backup 1  
    "sk-ant-your-key-3",  # Backup 2
]

SUBSCRIPTION_PLANS = {
    "starter": {
        "price": 49,
        "requests_per_day": 100,
        "features": ["basic_ai", "shopify_integration"]
    },
    "growth": {
        "price": 99, 
        "requests_per_day": 500,
        "features": ["basic_ai", "shopify_integration", "analytics", "automation"]
    },
    "pro": {
        "price": 149,
        "requests_per_day": 2000,
        "features": ["everything", "priority_support", "custom_integrations"]
    }
}

# ===== DATA MODELS =====

class MerchantAccount(BaseModel):
    """Merchant account information"""
    shop_domain: str
    shopify_token: str
    plan: str = "starter"
    status: str = "active"  # active, trial, suspended, cancelled
    trial_ends: Optional[datetime] = None
    created_at: datetime
    last_active: datetime
    total_requests_today: int = 0
    total_requests_month: int = 0

class AIRequest(BaseModel):
    """Individual AI request tracking"""
    merchant_id: str
    request_text: str
    response_text: str
    tokens_used: int
    cost_estimate: float
    timestamp: datetime
    success: bool = True

# ===== CORE SERVICES =====

class MerchantManager:
    """Manage merchant accounts and billing"""
    
    def __init__(self):
        self.db_path = "merchants.db"
        
    async def initialize(self):
        """Setup database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS merchants (
                    shop_domain TEXT PRIMARY KEY,
                    shopify_token TEXT NOT NULL,
                    plan TEXT DEFAULT 'starter',
                    status TEXT DEFAULT 'trial',
                    trial_ends TEXT,
                    created_at TEXT,
                    last_active TEXT,
                    total_requests_today INTEGER DEFAULT 0,
                    total_requests_month INTEGER DEFAULT 0
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ai_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant_id TEXT,
                    request_text TEXT,
                    response_text TEXT,
                    tokens_used INTEGER,
                    cost_estimate REAL,
                    timestamp TEXT,
                    success BOOLEAN DEFAULT 1
                )
            """)
            
            await db.commit()
    
    async def create_merchant(self, shop_domain: str, shopify_token: str) -> MerchantAccount:
        """Create new merchant account with 7-day trial"""
        merchant = MerchantAccount(
            shop_domain=shop_domain,
            shopify_token=shopify_token,
            plan="starter",
            status="trial",
            trial_ends=datetime.utcnow() + timedelta(days=7),
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO merchants 
                (shop_domain, shopify_token, plan, status, trial_ends, created_at, last_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                merchant.shop_domain, merchant.shopify_token, merchant.plan,
                merchant.status, merchant.trial_ends.isoformat(),
                merchant.created_at.isoformat(), merchant.last_active.isoformat()
            ))
            await db.commit()
        
        return merchant
    
    async def get_merchant(self, shop_domain: str) -> Optional[MerchantAccount]:
        """Get merchant by shop domain"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM merchants WHERE shop_domain = ?", (shop_domain,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return MerchantAccount(
                        shop_domain=row[0],
                        shopify_token=row[1], 
                        plan=row[2],
                        status=row[3],
                        trial_ends=datetime.fromisoformat(row[4]) if row[4] else None,
                        created_at=datetime.fromisoformat(row[5]),
                        last_active=datetime.fromisoformat(row[6]),
                        total_requests_today=row[7],
                        total_requests_month=row[8]
                    )
        return None
    
    async def check_usage_limits(self, merchant: MerchantAccount) -> bool:
        """Check if merchant can make more requests"""
        plan_limits = SUBSCRIPTION_PLANS[merchant.plan]
        
        # Check if trial expired
        if merchant.status == "trial" and merchant.trial_ends < datetime.utcnow():
            return False
            
        # Check daily limits
        if merchant.total_requests_today >= plan_limits["requests_per_day"]:
            return False
            
        return True
    
    async def log_request(self, merchant: MerchantAccount, request: AIRequest):
        """Log AI request and update usage counters"""
        async with aiosqlite.connect(self.db_path) as db:
            # Log the request
            await db.execute("""
                INSERT INTO ai_requests 
                (merchant_id, request_text, response_text, tokens_used, cost_estimate, timestamp, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                request.merchant_id, request.request_text, request.response_text,
                request.tokens_used, request.cost_estimate, 
                request.timestamp.isoformat(), request.success
            ))
            
            # Update usage counters
            await db.execute("""
                UPDATE merchants 
                SET total_requests_today = total_requests_today + 1,
                    total_requests_month = total_requests_month + 1,
                    last_active = ?
                WHERE shop_domain = ?
            """, (datetime.utcnow().isoformat(), merchant.shop_domain))
            
            await db.commit()

class ClaudeAPIManager:
    """Manage Claude API keys and rate limiting"""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.key_usage = {key: {"requests_this_minute": 0, "last_reset": time.time()} 
                         for key in api_keys}
        self.session = None
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def get_available_key(self) -> str:
        """Get an API key that's under rate limits"""
        current_time = time.time()
        
        for key in self.api_keys:
            key_stats = self.key_usage[key]
            
            # Reset counter if a minute has passed
            if current_time - key_stats["last_reset"] >= 60:
                key_stats["requests_this_minute"] = 0
                key_stats["last_reset"] = current_time
            
            # Check if this key is available (under 50 requests/minute)
            if key_stats["requests_this_minute"] < 45:  # Leave some buffer
                key_stats["requests_this_minute"] += 1
                return key
        
        # All keys are rate limited
        raise Exception("All API keys are rate limited. Please try again in a minute.")
    
    async def call_claude(self, messages: List[Dict], merchant_id: str) -> Dict[str, Any]:
        """Make Claude API call with automatic key rotation"""
        api_key = await self.get_available_key()
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1000,
            "messages": messages
        }
        
        async with self.session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "content": result["content"][0]["text"],
                    "tokens_used": result["usage"]["input_tokens"] + result["usage"]["output_tokens"],
                    "cost_estimate": self._estimate_cost(result["usage"])
                }
            else:
                error = await response.text()
                return {
                    "success": False,
                    "error": f"API Error {response.status}: {error}",
                    "tokens_used": 0,
                    "cost_estimate": 0
                }
    
    def _estimate_cost(self, usage: Dict) -> float:
        """Estimate cost based on token usage"""
        input_cost = usage["input_tokens"] * 0.000003  # $3 per 1M tokens
        output_cost = usage["output_tokens"] * 0.000015  # $15 per 1M tokens
        return input_cost + output_cost

class ShopifyConnector:
    """Connect to Shopify API for each merchant"""
    
    @staticmethod
    async def setup_session(merchant: MerchantAccount):
        """Setup Shopify session for merchant"""
        session = shopify.Session(
            merchant.shop_domain,
            "2024-01",
            merchant.shopify_token
        )
        shopify.ShopifyResource.activate_session(session)
        return session
    
    @staticmethod
    async def get_products(merchant: MerchantAccount, limit: int = 10) -> List[Dict]:
        """Get products from merchant's store"""
        await ShopifyConnector.setup_session(merchant)
        
        products = shopify.Product.find(limit=limit)
        return [
            {
                "id": product.id,
                "title": product.title,
                "handle": product.handle,
                "product_type": product.product_type,
                "vendor": product.vendor,
                "status": getattr(product, 'status', 'active'),
                "inventory": sum([variant.inventory_quantity or 0 for variant in product.variants])
            }
            for product in products
        ]
    
    @staticmethod
    async def get_orders(merchant: MerchantAccount, limit: int = 10) -> List[Dict]:
        """Get recent orders"""
        await ShopifyConnector.setup_session(merchant)
        
        orders = shopify.Order.find(limit=limit, status="any")
        return [
            {
                "id": order.id,
                "name": order.name,
                "total_price": str(order.total_price),
                "created_at": str(order.created_at),
                "customer": order.customer.first_name + " " + order.customer.last_name if order.customer else "Guest",
                "line_items_count": len(order.line_items)
            }
            for order in orders
        ]

# ===== MAIN APPLICATION =====

class ShopifyAISaaS:
    """Main SaaS application"""
    
    def __init__(self):
        self.merchant_manager = MerchantManager()
        self.claude_api = ClaudeAPIManager(CLAUDE_API_KEYS)
        self.shopify = ShopifyConnector()
        
    async def initialize(self):
        """Initialize all services"""
        await self.merchant_manager.initialize()
        await self.claude_api.initialize()
        logger.info("Shopify AI SaaS initialized")
    
    async def process_merchant_request(
        self, 
        shop_domain: str, 
        user_message: str,
        ctx: Optional[Context] = None
    ) -> Dict[str, Any]:
        """Process AI request from merchant"""
        
        # Get merchant account
        merchant = await self.merchant_manager.get_merchant(shop_domain)
        if not merchant:
            return {"error": "Merchant not found. Please reinstall the app."}
        
        # Check usage limits
        if not await self.merchant_manager.check_usage_limits(merchant):
            return {
                "error": "Usage limit reached. Please upgrade your plan or wait for reset.",
                "upgrade_url": f"/upgrade/{shop_domain}"
            }
        
        # Build context for Claude
        shopify_context = await self._build_shopify_context(merchant, user_message)
        
        messages = [
            {
                "role": "user", 
                "content": f"""You are an AI assistant for a Shopify store. Here's the context:

Store: {merchant.shop_domain}
Plan: {merchant.plan}

Recent store data:
{shopify_context}

User question: {user_message}

Please provide a helpful response about their Shopify store. If they're asking for specific data, use the context provided. If they want to make changes, explain what can be done and ask for confirmation."""
            }
        ]
        
        # Call Claude API
        try:
            if ctx:
                await ctx.info(f"Processing request for {shop_domain}")
                
            result = await self.claude_api.call_claude(messages, merchant.shop_domain)
            
            # Log the request
            ai_request = AIRequest(
                merchant_id=merchant.shop_domain,
                request_text=user_message,
                response_text=result.get("content", ""),
                tokens_used=result.get("tokens_used", 0),
                cost_estimate=result.get("cost_estimate", 0),
                timestamp=datetime.utcnow(),
                success=result.get("success", False)
            )
            
            await self.merchant_manager.log_request(merchant, ai_request)
            
            if result["success"]:
                return {
                    "success": True,
                    "response": result["content"],
                    "usage": {
                        "requests_today": merchant.total_requests_today + 1,
                        "plan_limit": SUBSCRIPTION_PLANS[merchant.plan]["requests_per_day"]
                    }
                }
            else:
                return {"error": result["error"]}
                
        except Exception as e:
            logger.error(f"Error processing request for {shop_domain}: {e}")
            return {"error": "Service temporarily unavailable. Please try again."}
    
    async def _build_shopify_context(self, merchant: MerchantAccount, user_message: str) -> str:
        """Build relevant Shopify context based on user question"""
        context_parts = []
        
        # Always include basic store info
        try:
            if "product" in user_message.lower() or "inventory" in user_message.lower():
                products = await self.shopify.get_products(merchant, limit=5)
                context_parts.append(f"Recent Products: {json.dumps(products, indent=2)}")
            
            if "order" in user_message.lower() or "sale" in user_message.lower():
                orders = await self.shopify.get_orders(merchant, limit=5)
                context_parts.append(f"Recent Orders: {json.dumps(orders, indent=2)}")
                
        except Exception as e:
            logger.error(f"Error building context for {merchant.shop_domain}: {e}")
            context_parts.append("Note: Unable to fetch current store data")
        
        return "\n\n".join(context_parts) if context_parts else "Store data unavailable"

# ===== FASTMCP SERVER =====

# Create global SaaS instance
saas_app = ShopifyAISaaS()

# Create FastMCP server
mcp = FastMCP(
    name="Shopify AI Assistant SaaS",
    dependencies=["aiohttp", "aiosqlite", "shopify", "pydantic"]
)

@mcp.tool
async def install_merchant(
    shop_domain: str,
    shopify_access_token: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Install the AI assistant for a new Shopify merchant.
    Called automatically when merchant installs the app.
    """
    try:
        await saas_app.initialize()
        
        if ctx:
            await ctx.info(f"Installing AI assistant for {shop_domain}")
        
        # Create merchant account with 7-day trial
        merchant = await saas_app.merchant_manager.create_merchant(
            shop_domain, shopify_access_token
        )
        
        return {
            "success": True,
            "message": f"🎉 AI Assistant installed for {shop_domain}!",
            "trial_ends": merchant.trial_ends.isoformat(),
            "plan": merchant.plan,
            "daily_limit": SUBSCRIPTION_PLANS[merchant.plan]["requests_per_day"]
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Installation failed: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool
async def chat_with_ai(
    shop_domain: str,
    message: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Main chat interface - merchants ask questions about their store.
    
    Args:
        shop_domain: The Shopify store domain (automatically provided)
        message: The merchant's question or request
    """
    try:
        await saas_app.initialize()
        
        result = await saas_app.process_merchant_request(
            shop_domain, message, ctx
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Chat error for {shop_domain}: {e}")
        return {
            "success": False,
            "error": "Something went wrong. Please try again or contact support."
        }

@mcp.tool
async def get_merchant_dashboard(
    shop_domain: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Get merchant dashboard with usage stats and account info.
    """
    try:
        await saas_app.initialize()
        
        merchant = await saas_app.merchant_manager.get_merchant(shop_domain)
        if not merchant:
            return {"error": "Merchant not found"}
        
        plan_info = SUBSCRIPTION_PLANS[merchant.plan]
        
        return {
            "success": True,
            "account": {
                "shop_domain": merchant.shop_domain,
                "plan": merchant.plan,
                "status": merchant.status,
                "trial_ends": merchant.trial_ends.isoformat() if merchant.trial_ends else None
            },
            "usage": {
                "requests_today": merchant.total_requests_today,
                "requests_month": merchant.total_requests_month,
                "daily_limit": plan_info["requests_per_day"],
                "plan_features": plan_info["features"]
            },
            "billing": {
                "monthly_price": plan_info["price"],
                "next_billing": "Auto-calculated based on install date"
            }
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Dashboard error: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool
async def upgrade_merchant_plan(
    shop_domain: str,
    new_plan: str,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Upgrade merchant to a different plan.
    
    Args:
        shop_domain: Store domain
        new_plan: starter, growth, or pro
    """
    try:
        if new_plan not in SUBSCRIPTION_PLANS:
            return {"error": f"Invalid plan: {new_plan}"}
        
        await saas_app.initialize()
        
        # Update merchant plan in database
        async with aiosqlite.connect(saas_app.merchant_manager.db_path) as db:
            await db.execute(
                "UPDATE merchants SET plan = ?, status = 'active' WHERE shop_domain = ?",
                (new_plan, shop_domain)
            )
            await db.commit()
        
        plan_info = SUBSCRIPTION_PLANS[new_plan]
        
        if ctx:
            await ctx.info(f"Upgraded {shop_domain} to {new_plan} plan")
        
        return {
            "success": True,
            "message": f"🚀 Upgraded to {new_plan.title()} plan!",
            "new_features": plan_info["features"],
            "daily_limit": plan_info["requests_per_day"],
            "monthly_price": plan_info["price"]
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Upgrade failed: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool
async def get_saas_analytics(ctx: Context = None) -> Dict[str, Any]:
    """
    Get SaaS business analytics (for you, the app owner).
    """
    try:
        await saas_app.initialize()
        
        async with aiosqlite.connect(saas_app.merchant_manager.db_path) as db:
            # Get merchant counts by plan
            async with db.execute("""
                SELECT plan, COUNT(*) as count, status
                FROM merchants 
                GROUP BY plan, status
            """) as cursor:
                plan_stats = await cursor.fetchall()
            
            # Get total revenue estimate
            total_revenue = 0
            total_merchants = 0
            for plan, count, status in plan_stats:
                if status == "active":
                    total_revenue += SUBSCRIPTION_PLANS[plan]["price"] * count
                    total_merchants += count
            
            # Get usage stats
            async with db.execute("""
                SELECT 
                    COUNT(*) as total_requests_today,
                    SUM(cost_estimate) as total_api_costs_today,
                    AVG(cost_estimate) as avg_cost_per_request
                FROM ai_requests 
                WHERE date(timestamp) = date('now')
            """) as cursor:
                usage_stats = await cursor.fetchone()
        
        return {
            "success": True,
            "business_metrics": {
                "total_merchants": total_merchants,
                "monthly_revenue": total_revenue,
                "plan_distribution": {
                    f"{plan}_{status}": count 
                    for plan, count, status in plan_stats
                }
            },
            "usage_metrics": {
                "requests_today": usage_stats[0] or 0,
                "api_costs_today": round(usage_stats[1] or 0, 4),
                "avg_cost_per_request": round(usage_stats[2] or 0, 6)
            },
            "profitability": {
                "revenue_per_merchant": total_revenue / max(total_merchants, 1),
                "estimated_profit_margin": "85-95%"
            }
        }
        
    except Exception as e:
        if ctx:
            await ctx.error(f"Analytics error: {str(e)}")
        return {"success": False, "error": str(e)}

# Cleanup on shutdown
async def cleanup():
    """Cleanup resources"""
    if saas_app.claude_api.session:
        await saas_app.claude_api.session.close()

import atexit
atexit.register(lambda: asyncio.create_task(cleanup()))

if __name__ == "__main__":
    mcp.run()