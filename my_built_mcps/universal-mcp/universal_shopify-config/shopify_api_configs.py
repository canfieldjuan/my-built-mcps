#!/usr/bin/env python3
# File: shopify_api_configs.py
"""
Shopify API Pre-Configurations for Universal API Gateway
Includes both REST (legacy) and GraphQL configurations
"""

import asyncio
import json
from pathlib import Path

# Shopify API Configurations
SHOPIFY_CONFIGS = {
    "services": {
        "shopify_rest_admin": {
            "name": "shopify_rest_admin",
            "base_url": "https://{shop_name}.myshopify.com/admin/api/2024-04",
            "auth_type": "api_key",
            "auth_config": {
                "location": "header",
                "key": "X-Shopify-Access-Token",
                "value": "{SHOPIFY_ACCESS_TOKEN}"  # Replace with actual token
            },
            "default_headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "rate_limit": 2000,  # 2000 requests per minute (Shopify Plus limit)
            "timeout": 30,
            "retry_attempts": 3,
            "cache_ttl": 300,
            "circuit_breaker_enabled": True,
            "health_check_path": "/shop.json"
        },
        
        "shopify_graphql_admin": {
            "name": "shopify_graphql_admin", 
            "base_url": "https://{shop_name}.myshopify.com/admin/api/2024-04",
            "auth_type": "api_key",
            "auth_config": {
                "location": "header",
                "key": "X-Shopify-Access-Token", 
                "value": "{SHOPIFY_ACCESS_TOKEN}"  # Replace with actual token
            },
            "default_headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "rate_limit": 1000,  # GraphQL typically has different limits
            "timeout": 45,  # GraphQL queries can be more complex
            "retry_attempts": 3,
            "cache_ttl": 180,  # Shorter cache for dynamic data
            "circuit_breaker_enabled": True,
            "health_check_path": "/shop.json"
        },
        
        "shopify_storefront": {
            "name": "shopify_storefront",
            "base_url": "https://{shop_name}.myshopify.com/api/2024-04",
            "auth_type": "api_key",
            "auth_config": {
                "location": "header",
                "key": "X-Shopify-Storefront-Access-Token",
                "value": "{SHOPIFY_STOREFRONT_TOKEN}"  # Replace with storefront token
            },
            "default_headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "rate_limit": 500,  # Storefront API has different limits
            "timeout": 30,
            "retry_attempts": 3,
            "cache_ttl": 600,  # Longer cache for storefront data
            "circuit_breaker_enabled": True
        }
    },
    
    "endpoints": {
        # ===== REST ADMIN API ENDPOINTS (Legacy - being deprecated) =====
        
        # Shop Information
        "shopify_get_shop": {
            "name": "shopify_get_shop",
            "service": "shopify_rest_admin",
            "path": "/shop.json",
            "method": "GET", 
            "description": "Get shop information and settings",
            "parameters_schema": {},
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["shop", "admin", "rest"]
        },
        
        # Products - Use GraphQL only (REST endpoints no longer available)
        # NOTE: Product/Variant REST endpoints were discontinued February 2025
        
        # Orders
        "shopify_list_orders": {
            "name": "shopify_list_orders",
            "service": "shopify_rest_admin",
            "path": "/orders.json",
            "method": "GET",
            "description": "List all orders with filtering options",
            "parameters_schema": {
                "type": "object", 
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 250, "default": 50},
                    "since_id": {"type": "integer"},
                    "created_at_min": {"type": "string", "format": "date-time"},
                    "created_at_max": {"type": "string", "format": "date-time"},
                    "updated_at_min": {"type": "string", "format": "date-time"},
                    "updated_at_max": {"type": "string", "format": "date-time"},
                    "processed_at_min": {"type": "string", "format": "date-time"},
                    "processed_at_max": {"type": "string", "format": "date-time"},
                    "status": {"type": "string", "enum": ["open", "closed", "cancelled", "any"]},
                    "financial_status": {"type": "string", "enum": ["pending", "authorized", "partially_paid", "paid", "partially_refunded", "refunded", "voided"]},
                    "fulfillment_status": {"type": "string", "enum": ["shipped", "partial", "unshipped", "any"]},
                    "fields": {"type": "string"}
                }
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["orders", "admin", "rest"]
        },
        
        "shopify_get_order": {
            "name": "shopify_get_order",
            "service": "shopify_rest_admin",
            "path": "/orders/{order_id}.json",
            "method": "GET",
            "description": "Get a specific order by ID",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "Order ID"},
                    "fields": {"type": "string"}
                },
                "required": ["order_id"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["orders", "admin", "rest"]
        },
        
        # Customers
        "shopify_list_customers": {
            "name": "shopify_list_customers",
            "service": "shopify_rest_admin",
            "path": "/customers.json",
            "method": "GET",
            "description": "List all customers",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 250, "default": 50},
                    "since_id": {"type": "integer"},
                    "created_at_min": {"type": "string", "format": "date-time"},
                    "created_at_max": {"type": "string", "format": "date-time"},
                    "updated_at_min": {"type": "string", "format": "date-time"},
                    "updated_at_max": {"type": "string", "format": "date-time"},
                    "fields": {"type": "string"}
                }
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["customers", "admin", "rest"]
        },
        
        "shopify_get_customer": {
            "name": "shopify_get_customer",
            "service": "shopify_rest_admin",
            "path": "/customers/{customer_id}.json",
            "method": "GET",
            "description": "Get a specific customer by ID",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer", "description": "Customer ID"},
                    "fields": {"type": "string"}
                },
                "required": ["customer_id"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["customers", "admin", "rest"]
        },
        
        # Inventory
        "shopify_list_inventory_levels": {
            "name": "shopify_list_inventory_levels",
            "service": "shopify_rest_admin",
            "path": "/inventory_levels.json",
            "method": "GET",
            "description": "List inventory levels",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "inventory_item_ids": {"type": "string", "description": "Comma-separated inventory item IDs"},
                    "location_ids": {"type": "string", "description": "Comma-separated location IDs"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 250, "default": 50},
                    "updated_at_min": {"type": "string", "format": "date-time"}
                }
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["inventory", "admin", "rest"]
        },
        
        # Collections
        "shopify_list_collections": {
            "name": "shopify_list_collections",
            "service": "shopify_rest_admin",
            "path": "/custom_collections.json",
            "method": "GET",
            "description": "List custom collections",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 250, "default": 50},
                    "since_id": {"type": "integer"},
                    "created_at_min": {"type": "string", "format": "date-time"},
                    "created_at_max": {"type": "string", "format": "date-time"},
                    "updated_at_min": {"type": "string", "format": "date-time"},
                    "updated_at_max": {"type": "string", "format": "date-time"},
                    "published_status": {"type": "string", "enum": ["published", "unpublished", "any"]},
                    "fields": {"type": "string"}
                }
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["collections", "admin", "rest"]
        },
        
        # ===== GRAPHQL ADMIN API ENDPOINTS (Recommended) =====
        
        "shopify_graphql_products": {
            "name": "shopify_graphql_products",
            "service": "shopify_graphql_admin",
            "path": "/graphql.json",
            "method": "POST",
            "description": "Query products using GraphQL (REQUIRED - REST product endpoints discontinued)",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "GraphQL query string",
                        "default": "query getProducts($first: Int!) { products(first: $first) { edges { node { id title handle status createdAt updatedAt vendor productType tags variants(first: 10) { edges { node { id title price sku inventory { available } } } } images(first: 5) { edges { node { id url altText } } } } } } }"
                    },
                    "variables": {
                        "type": "object",
                        "description": "GraphQL variables",
                        "default": {"first": 10}
                    }
                },
                "required": ["query"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["products", "admin", "graphql", "required"]
        },
        
        "shopify_graphql_orders": {
            "name": "shopify_graphql_orders",
            "service": "shopify_graphql_admin",
            "path": "/graphql.json",
            "method": "POST",
            "description": "Query orders using GraphQL",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "GraphQL query string",
                        "default": "query getOrders($first: Int!) { orders(first: $first) { edges { node { id name email createdAt updatedAt totalPriceSet { shopMoney { amount currencyCode } } customer { id firstName lastName email } lineItems(first: 10) { edges { node { title quantity currentQuantity variant { id title price sku } } } } shippingAddress { address1 city province country zip } } } } }"
                    },
                    "variables": {
                        "type": "object",
                        "description": "GraphQL variables",
                        "default": {"first": 10}
                    }
                },
                "required": ["query"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["orders", "admin", "graphql"]
        },
        
        "shopify_graphql_customers": {
            "name": "shopify_graphql_customers",
            "service": "shopify_graphql_admin", 
            "path": "/graphql.json",
            "method": "POST",
            "description": "Query customers using GraphQL",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "GraphQL query string",
                        "default": "query getCustomers($first: Int!) { customers(first: $first) { edges { node { id firstName lastName email phone createdAt updatedAt acceptsMarketing ordersCount totalSpent addresses { address1 city province country zip } } } } }"
                    },
                    "variables": {
                        "type": "object",
                        "description": "GraphQL variables", 
                        "default": {"first": 10}
                    }
                },
                "required": ["query"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["customers", "admin", "graphql"]
        },
        
        "shopify_graphql_create_product": {
            "name": "shopify_graphql_create_product",
            "service": "shopify_graphql_admin",
            "path": "/graphql.json", 
            "method": "POST",
            "description": "Create a product using GraphQL (REQUIRED - REST product creation discontinued)",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "GraphQL mutation string",
                        "default": "mutation productCreate($input: ProductInput!) { productCreate(input: $input) { product { id title handle status vendor productType } userErrors { field message } } }"
                    },
                    "variables": {
                        "type": "object",
                        "description": "Product input variables",
                        "properties": {
                            "input": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "descriptionHtml": {"type": "string"},
                                    "vendor": {"type": "string"},
                                    "productType": {"type": "string"},
                                    "status": {"type": "string", "enum": ["ACTIVE", "ARCHIVED", "DRAFT"]},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                    "variants": {"type": "array"}
                                },
                                "required": ["title"]
                            }
                        },
                        "required": ["input"]
                    }
                },
                "required": ["query", "variables"]
            },
            "cache_enabled": False,
            "requires_auth": True,
            "tags": ["products", "admin", "graphql", "mutation"]
        },
        
        # ===== STOREFRONT API ENDPOINTS =====
        
        "shopify_storefront_products": {
            "name": "shopify_storefront_products",
            "service": "shopify_storefront",
            "path": "/graphql.json",
            "method": "POST",
            "description": "Query public products for storefront",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "GraphQL query string",
                        "default": "query getProducts($first: Int!) { products(first: $first) { edges { node { id title handle description availableForSale vendor productType tags priceRange { minVariantPrice { amount currencyCode } maxVariantPrice { amount currencyCode } } images(first: 5) { edges { node { id url altText } } } variants(first: 10) { edges { node { id title price { amount currencyCode } selectedOptions { name value } availableForSale } } } } } } }"
                    },
                    "variables": {
                        "type": "object",
                        "description": "GraphQL variables",
                        "default": {"first": 10}
                    }
                },
                "required": ["query"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["products", "storefront", "graphql", "public"]
        },
        
        "shopify_storefront_collections": {
            "name": "shopify_storefront_collections",
            "service": "shopify_storefront",
            "path": "/graphql.json",
            "method": "POST",
            "description": "Query public collections for storefront",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "GraphQL query string",
                        "default": "query getCollections($first: Int!) { collections(first: $first) { edges { node { id title handle description image { id url altText } products(first: 10) { edges { node { id title handle priceRange { minVariantPrice { amount currencyCode } } } } } } } } }"
                    },
                    "variables": {
                        "type": "object",
                        "description": "GraphQL variables",
                        "default": {"first": 10}
                    }
                },
                "required": ["query"]
            },
            "cache_enabled": True,
            "requires_auth": True,
            "tags": ["collections", "storefront", "graphql", "public"]
        }
    }
}

# Configuration Setup Script
async def setup_shopify_configurations(gateway):
    """Setup Shopify API configurations in the Universal API Gateway"""
    
    print("🛍️ Setting up Shopify API configurations...")
    
    try:
        # Ensure gateway is initialized
        await gateway.initialize()
        
        # Configure services
        services_added = 0
        for service_name, service_config in SHOPIFY_CONFIGS["services"].items():
            try:
                gateway.services[service_name] = APIServiceConfig(**service_config)
                services_added += 1
                print(f"✅ Added service: {service_name}")
            except Exception as e:
                print(f"❌ Failed to add service {service_name}: {e}")
        
        # Configure endpoints
        endpoints_added = 0
        for endpoint_name, endpoint_config in SHOPIFY_CONFIGS["endpoints"].items():
            try:
                gateway.endpoints[endpoint_name] = EndpointConfig(**endpoint_config)
                endpoints_added += 1
                print(f"✅ Added endpoint: {endpoint_name}")
            except Exception as e:
                print(f"❌ Failed to add endpoint {endpoint_name}: {e}")
        
        # Save configurations
        await gateway.save_configurations()
        
        print(f"""
🎉 Shopify API configuration complete!

📊 Summary:
   • Services added: {services_added}/3
   • Endpoints added: {endpoints_added}
   
🔐 Next Steps:
   1. Replace {'{shop_name}'} with your actual shop name (e.g., 'mystore')
   2. Replace {'{SHOPIFY_ACCESS_TOKEN}'} with your Admin API access token
   3. Replace {'{SHOPIFY_STOREFRONT_TOKEN}'} with your Storefront API token (if using)
   
📚 Available Endpoints:
   REST Admin (Legacy):
   • Products, Orders, Customers, Inventory, Collections
   
   GraphQL Admin (Recommended):  
   • Products, Orders, Customers + Mutations
   
   Storefront API:
   • Public products and collections
   
⚠️  Important Notes:
   • REST Product/Variant endpoints deprecated Feb 1, 2025
   • Use GraphQL for new integrations
   • Rate limits: Admin=2000/min, Storefront=500/min
        """)
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

# Environment Configuration Template
def generate_env_template():
    """Generate environment variable template for Shopify"""
    template = """
# Shopify API Configuration
# Replace these values with your actual credentials

# Your Shopify store name (without .myshopify.com)
SHOPIFY_SHOP_NAME=your-store-name

# Admin API Access Token (from Custom App in Shopify Admin)
SHOPIFY_ACCESS_TOKEN=shpat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Storefront API Access Token (for public data access)
SHOPIFY_STOREFRONT_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# API Version (current stable version)
SHOPIFY_API_VERSION=2024-04

# Rate Limiting (requests per minute)
SHOPIFY_ADMIN_RATE_LIMIT=2000
SHOPIFY_STOREFRONT_RATE_LIMIT=500
"""
    
    env_file = Path(".env.shopify")
    with open(env_file, "w") as f:
        f.write(template)
    
    print(f"📝 Environment template created: {env_file}")
    print("   Please update with your actual Shopify credentials")

# Example Usage Functions
def get_example_usage():
    """Return example API calls for testing"""
    examples = {
        "get_shop_info": {
            "service": "shopify_rest_admin",
            "endpoint": "shopify_get_shop",
            "description": "Get basic shop information"
        },
        
        "list_products_rest": {
            "service": "shopify_rest_admin", 
            "endpoint": "shopify_list_products",
            "params": {"limit": 10, "published_status": "published"},
            "description": "List products using REST (legacy)"
        },
        
        "list_products_graphql": {
            "service": "shopify_graphql_admin",
            "endpoint": "shopify_graphql_products", 
            "body": {
                "query": "query { products(first: 5) { edges { node { id title handle vendor } } } }"
            },
            "description": "List products using GraphQL (recommended)"
        },
        
        "get_orders": {
            "service": "shopify_rest_admin",
            "endpoint": "shopify_list_orders",
            "params": {"limit": 5, "status": "open"},
            "description": "Get recent open orders"
        },
        
        "storefront_products": {
            "service": "shopify_storefront",
            "endpoint": "shopify_storefront_products",
            "body": {
                "query": "query { products(first: 3) { edges { node { id title handle price { amount } } } } }"
            },
            "description": "Get public products for storefront"
        }
    }
    
    return examples

if __name__ == "__main__":
    print("Shopify API Gateway Configuration")
    print("="*50)
    
    # Generate environment template
    generate_env_template()
    
    # Show example usage
    examples = get_example_usage()
    print("\n📋 Example API Calls:")
    for name, example in examples.items():
        print(f"\n{name}:")
        print(f"  Service: {example['service']}")
        print(f"  Endpoint: {example['endpoint']}")
        print(f"  Description: {example['description']}")
        if 'params' in example:
            print(f"  Params: {example['params']}")
        if 'body' in example:
            print(f"  Body: {example['body']}")