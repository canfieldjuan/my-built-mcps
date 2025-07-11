# 🔐 Shopify Credentials Setup Guide
<!-- File: SHOPIFY_SETUP.md -->

## Step-by-Step Authentication Setup

### Method 1: Custom App in Shopify Admin (Recommended)

1. **Access Shopify Admin**
   - Log into your Shopify store admin panel
   - Go to `Settings` → `Apps and sales channels`

2. **Create Custom App**
   - Click `Develop apps`
   - Click `Create an app`
   - Enter app name (e.g., "API Gateway Integration")
   - Select developer (yourself)

3. **Configure API Scopes**
   Click `Configure Admin API scopes` and enable:
   
   **Essential Scopes:**
   - ✅ `read_products` - View products
   - ✅ `write_products` - Create/update products (if needed)
   - ✅ `read_orders` - View orders
   - ✅ `read_customers` - View customers
   - ✅ `read_inventory` - Check inventory levels
   
   **Optional Scopes:**
   - ⚪ `write_orders` - Create/modify orders
   - ⚪ `write_customers` - Create/modify customers
   - ⚪ `write_inventory` - Update inventory
   - ⚪ `read_analytics` - Access analytics data
   - ⚪ `read_all_orders` - Access all historical orders

4. **Install and Get Token**
   - Click `Install app`
   - Copy the `Admin API access token` (starts with `shpat_`)
   - ⚠️ **Important**: You can only view this token once!

### Method 2: Private App (Legacy)

If your store still supports private apps:

1. Go to `Apps` → `Manage private apps`
2. Create private app
3. Enable required API permissions
4. Copy the password (this is your access token)

## 🏪 Configuration Values You Need

### Required Information
```bash
# Your store details
SHOP_NAME="your-store-name"  # Without .myshopify.com
SHOP_URL="https://your-store-name.myshopify.com"

# API credentials
SHOPIFY_ACCESS_TOKEN="shpat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Optional: Storefront API (for public data)
SHOPIFY_STOREFRONT_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Rate Limits by Plan
- **Shopify Plus**: 2000 requests/minute
- **Advanced Shopify**: 1000 requests/minute  
- **Shopify**: 500 requests/minute
- **Basic Shopify**: 500 requests/minute

## 🔧 Environment Setup

### Option 1: Environment Variables
```bash
# Add to your .env file
SHOPIFY_SHOP_NAME=your-store-name
SHOPIFY_ACCESS_TOKEN=shpat_your_actual_token_here
SHOPIFY_API_VERSION=2024-04
SHOPIFY_RATE_LIMIT=2000
```

### Option 2: Direct Configuration
Replace placeholders in the service configurations:
- `{shop_name}` → your actual store name
- `{SHOPIFY_ACCESS_TOKEN}` → your actual access token

## 🧪 Testing Your Setup

### 1. Test Basic Connectivity
```python
# Test shop info endpoint
result = await api_request(
    service="shopify_rest_admin",
    endpoint="shopify_get_shop"
)

if result["status"] == "success":
    shop_name = result["response"]["data"]["shop"]["name"]
    print(f"✅ Connected to: {shop_name}")
else:
    print(f"❌ Connection failed: {result['error']}")
```

### 2. Test Authentication
```python
# This should return 401 if auth fails
# Should return shop data if auth succeeds
```

### 3. Test Rate Limiting
```python
# Make multiple rapid requests to test rate limiting
for i in range(5):
    result = await api_request(
        service="shopify_rest_admin", 
        endpoint="shopify_get_shop"
    )
    print(f"Request {i+1}: {result['status']}")
```

## 🚨 Security Best Practices

### ✅ Do's
- Store tokens in environment variables
- Use minimum required scopes
- Rotate tokens periodically
- Monitor API usage in Shopify admin
- Use HTTPS only
- Implement proper error handling

### ❌ Don'ts  
- Don't hardcode tokens in source code
- Don't commit tokens to version control
- Don't share tokens publicly
- Don't request excessive scopes
- Don't ignore rate limits

## 🔄 Token Management

### Rotating Tokens
1. Create new custom app
2. Update configuration with new token
3. Test thoroughly
4. Delete old app

### Monitoring Usage
- Check `Analytics` → `Reports` in Shopify admin
- Monitor rate limit headers in API responses
- Set up alerts for high usage

## ⚠️ Current API Status

### Product/Variant Operations
- **REST Endpoints**: DISCONTINUED (February 2025)
- **GraphQL Required**: ALL product operations must use GraphQL
- **No Migration Needed**: Already happened

### Other Endpoints
- **REST Still Available**: Orders, customers, shop info, etc.
- **GraphQL Recommended**: Better performance and future-proof
- **Gradual Deprecation**: Other REST endpoints may be deprecated over time

### GraphQL Benefits
- More efficient data fetching
- Stronger type system
- Future-proof (Shopify's focus)
- Better performance for complex queries

## 🆘 Troubleshooting

### Common Issues

**401 Unauthorized**
- Check if token is correct
- Verify scopes are sufficient
- Ensure token hasn't expired

**403 Forbidden** 
- App doesn't have required permissions
- API scope not granted during app installation

**429 Rate Limited**
- Reduce request frequency
- Implement exponential backoff
- Check your Shopify plan limits

**404 Not Found**
- Verify shop name is correct
- Check API version in URL
- Ensure endpoint path is valid

### Debug Steps
1. Test with Shopify's GraphiQL tool
2. Check API call logs in Shopify admin
3. Verify authentication headers
4. Test with curl/Postman first
5. Check circuit breaker status

## 📚 Additional Resources

- [Shopify Admin API Reference](https://shopify.dev/docs/api/admin-rest)
- [GraphQL Admin API](https://shopify.dev/docs/api/admin-graphql) 
- [Authentication Guide](https://shopify.dev/docs/apps/auth)
- [Rate Limiting](https://shopify.dev/docs/api/usage/rate-limits)
- [API Versioning](https://shopify.dev/docs/api/usage/versioning)

Your Universal API Gateway is now ready to connect to Shopify! 🎉