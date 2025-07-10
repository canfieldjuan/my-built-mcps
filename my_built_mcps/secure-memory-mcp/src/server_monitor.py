"""
Server Monitor - Watches the MCP server for anomalies
Runs independently of the main server
"""

import asyncio
import psutil
import time
from pathlib import Path
from datetime import datetime
import json

class ServerMonitor:
    def __init__(self, server_pid: int, log_path: Path):
        self.server_pid = server_pid
        self.log_path = log_path
        self.anomalies = []
        
    async def monitor(self):
        """Monitor server health and detect anomalies"""
        process = psutil.Process(self.server_pid)
        
        while True:
            try:
                # Check CPU usage
                cpu_percent = process.cpu_percent(interval=1.0)
                if cpu_percent > 80:
                    self.log_anomaly("High CPU usage", {"cpu_percent": cpu_percent})
                
                # Check memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                if memory_mb > 500:  # 500MB threshold
                    self.log_anomaly("High memory usage", {"memory_mb": memory_mb})
                
                # Check file descriptors
                num_fds = process.num_fds()
                if num_fds > 100:
                    self.log_anomaly("High file descriptor count", {"fd_count": num_fds})
                
                # Check for suspicious child processes
                children = process.children()
                if children:
                    self.log_anomaly("Unexpected child processes", {
                        "children": [p.name() for p in children]
                    })
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except psutil.NoSuchProcess:
                self.log_anomaly("Server process terminated", {})
                break
            except Exception as e:
                self.log_anomaly("Monitor error", {"error": str(e)})
    
    def log_anomaly(self, anomaly_type: str, details: dict):
        anomaly = {
            "timestamp": datetime.now().isoformat(),
            "type": anomaly_type,
            "details": details
        }
        self.anomalies.append(anomaly)
        
        # Write to log file
        with open(self.log_path / "anomalies.json", "a") as f:
            f.write(json.dumps(anomaly) + "\n")
        
        # Could trigger alerts here
        print(f"ANOMALY DETECTED: {anomaly_type} - {details}")
