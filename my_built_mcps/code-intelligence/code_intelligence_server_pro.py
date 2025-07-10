{
  "federation_config": {
    "name": "Juan's MCP Federation",
    "version": "1.0.0",
    "health_check_interval": 30,
    "auto_recovery": true,
    "load_balancing": true,
    "logging_level": "INFO"
  },
  "toolkits": {
    "database": {
      "name": "Database Operations",
      "description": "All database operations across MySQL, SQLite, and Supabase",
      "namespace": "db",
      "priority": 1,
      "servers": [
        {
          "name": "mysql",
          "role": "primary",
          "weight": 100,
          "capabilities": ["query", "insert", "update", "delete", "schema"],
          "config": {
            "command": "npx",
            "args": ["-y", "@benborla29/mcp-server-mysql"],
            "env": {
              "MYSQL_HOST": "127.0.0.1",
              "MYSQL_PORT": "3306",
              "MYSQL_USER": "root",
              "MYSQL_PASS": "@Canfi1287",
              "MYSQL_DB": "big_picture_project",
              "ALLOW_INSERT_OPERATION": "true",
              "ALLOW_UPDATE_OPERATION": "true",
              "ALLOW_DELETE_OPERATION": "true"
            }
          }
        },
        {
          "name": "sqlite",
          "role": "fallback",
          "weight": 80,
          "capabilities": ["query", "insert", "update", "delete", "schema"],
          "config": {
            "command": "npx",
            "args": ["-y", "mcp-server-sqlite-npx", "C:/Users/Juan/AppData/Roaming/Claude/projects/local_data.db"]
          }
        },
        {
          "name": "supabase",
          "role": "fallback",
          "weight": 60,
          "capabilities": ["query", "insert", "update", "delete", "realtime"],
          "config": {
            "command": "npx",
            "args": ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", "<personal-access-token>"]
          }
        }
      ],
      "load_balancing": {
        "strategy": "weighted",
        "health_check": true,
        "failover": true
      }
    },
    "web_automation": {
      "name": "Web Automation & Scraping",
      "description": "Browser automation, web scraping, and page interaction",
      "namespace": "web",
      "priority": 2,
      "servers": [
        {
          "name": "playwright",
          "role": "primary",
          "weight": 100,
          "capabilities": ["navigate", "click", "extract", "screenshot", "forms", "modern_js"],
          "config": {
            "command": "npx",
            "args": ["@playwright/mcp@latest"]
          }
        },
        {
          "name": "puppeteer",
          "role": "fallback",
          "weight": 85,
          "capabilities": ["navigate", "click", "extract", "screenshot", "forms"],
          "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
          }
        },
        {
          "name": "browsermcp",
          "role": "fallback",
          "weight": 70,
          "capabilities": ["navigate", "extract", "simple_interactions"],
          "config": {
            "command": "npx",
            "args": ["@browsermcp/mcp@latest"]
          }
        }
      ],
      "load_balancing": {
        "strategy": "capability_based",
        "health_check": true,
        "failover": true
      }
    },
    "file_operations": {
      "name": "File Processing & Generation",
      "description": "Excel, PDF, and document operations",
      "namespace": "file",
      "priority": 3,
      "servers": [
        {
          "name": "excel",
          "role": "primary",
          "weight": 100,
          "capabilities": ["read_excel", "write_excel", "data_analysis", "charts"],
          "config": {
            "command": "cmd",
            "args": ["/c", "npx", "--yes", "@negokaz/excel-mcp-server"],
            "env": {
              "EXCEL_MCP_PAGING_CELLS_LIMIT": "4000"
            }
          }
        },
        {
          "name": "pdf-generator",
          "role": "secondary",
          "weight": 90,
          "capabilities": ["generate_pdf", "html_to_pdf", "document_conversion"],
          "config": {
            "command": "npx",
            "args": ["github:FabianGenell/pdf-mcp-server"]
          }
        }
      ],
      "load_balancing": {
        "strategy": "capability_based",
        "health_check": true,
        "failover": false
      }
    },
    "development": {
      "name": "Development & Code Analysis",
      "description": "GitHub operations and intelligent code analysis",
      "namespace": "dev",
      "priority": 4,
      "servers": [
        {
          "name": "code-intelligence-pro",
          "role": "primary",
          "weight": 100,
          "capabilities": ["code_analysis", "architecture", "refactoring", "expert_patterns", "cost_optimization"],
          "config": {
            "command": "python",
            "args": ["code_intelligence_server_pro.py"],
            "cwd": "C:\\Users\\Juan\\OneDrive\\Desktop\\claude_desktop_projects\\seo_auditor_production_clean"
          }
        },
        {
          "name": "github",
          "role": "secondary",
          "weight": 80,
          "capabilities": ["repo_operations", "issues", "prs", "releases"],
          "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
              "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
            }
          }
        }
      ],
      "load_balancing": {
        "strategy": "capability_based",
        "health_check": true,
        "failover": false
      }
    },
    "system": {
      "name": "System Operations & Memory",
      "description": "Desktop automation and memory management",
      "namespace": "sys",
      "priority": 5,
      "servers": [
        {
          "name": "desktop-commander",
          "role": "primary",
          "weight": 100,
          "capabilities": ["system_commands", "app_control", "file_system", "automation"],
          "config": {
            "command": "npx.cmd",
            "args": ["@wonderwhy-er/desktop-commander@latest"]
          }
        },
        {
          "name": "memory",
          "role": "secondary",
          "weight": 90,
          "capabilities": ["memory_storage", "context_retention", "recall"],
          "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"]
          }
        }
      ],
      "load_balancing": {
        "strategy": "capability_based",
        "health_check": true,
        "failover": true
      }
    },
    "ai_enhancement": {
      "name": "AI Reasoning Enhancement",
      "description": "Advanced reasoning and thinking tools",
      "namespace": "ai",
      "priority": 6,
      "servers": [
        {
          "name": "sequential-thinking",
          "role": "primary",
          "weight": 100,
          "capabilities": ["step_by_step_reasoning", "analysis", "problem_solving"],
          "config": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
          }
        }
      ],
      "load_balancing": {
        "strategy": "single_server",
        "health_check": true,
        "failover": false
      }
    }
  },
  "routing_rules": {
    "default_timeout": 30,
    "retry_attempts": 2,
    "circuit_breaker": {
      "enabled": true,
      "failure_threshold": 5,
      "timeout": 60
    },
    "load_balancing_strategies": {
      "weighted": "Route based on server weights and health",
      "capability_based": "Route based on specific capabilities needed",
      "round_robin": "Distribute requests evenly across servers",
      "single_server": "Use only the primary server"
    }
  },
  "monitoring": {
    "health_checks": {
      "enabled": true,
      "interval": 30,
      "timeout": 10,
      "retries": 2
    },
    "metrics": {
      "response_times": true,
      "error_rates": true,
      "server_utilization": true,
      "toolkit_usage": true
    },
    "alerts": {
      "server_down": true,
      "high_error_rate": true,
      "slow_response": true,
      "circuit_breaker_open": true
    }
  },
  "features": {
    "auto_discovery": {
      "enabled": true,
      "scan_interval": 300,
      "description": "Automatically discover new MCP servers"
    },
    "caching": {
      "enabled": true,
      "ttl": 300,
      "description": "Cache tool responses for better performance"
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100,
      "description": "Prevent server overload"
    },
    "request_deduplication": {
      "enabled": true,
      "window": 5,
      "description": "Deduplicate identical concurrent requests"
    }
  },
  "security": {
    "authentication": {
      "enabled": false,
      "method": "token",
      "description": "Authentication for server access"
    },
    "authorization": {
      "enabled": false,
      "method": "rbac",
      "description": "Role-based access control"
    },
    "encryption": {
      "enabled": false,
      "method": "tls",
      "description": "Encrypt communication between servers"
    }
  }
}