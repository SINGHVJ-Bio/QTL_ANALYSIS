#!/usr/bin/env python3
"""
Performance monitoring utilities for QTL pipeline
Author: Dr. Vijay Singh
Email: vijay.s.gautam@gmail.com
"""

import psutil
import threading
import time
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger('QTLPipeline')

class PerformanceMonitor:
    def __init__(self, interval=60, log_file=None):
        self.interval = interval
        self.log_file = log_file
        self.monitor_thread = None
        self.monitor_data = {
            'start_time': datetime.now(),
            'max_cpu': 0,
            'max_memory': 0,
            'samples': []
        }
        self.is_monitoring = False
    
    def start(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            logger.warning("⚠️ Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, name="PerformanceMonitor")
        self.monitor_thread.start()
        logger.info(f"🔍 Started performance monitoring (interval: {self.interval}s)")
        
        return self.monitor_data
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update statistics
                self.monitor_data['max_cpu'] = max(self.monitor_data['max_cpu'], cpu_percent)
                self.monitor_data['max_memory'] = max(self.monitor_data['max_memory'], memory.percent)
                
                # Store sample
                sample = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_used_percent': disk.percent
                }
                self.monitor_data['samples'].append(sample)
                
                # Log current status
                logger.info(f"📊 Performance: CPU {cpu_percent:.1f}%, "
                           f"Memory {memory.percent:.1f}% ({memory.used/(1024**3):.1f}GB used), "
                           f"Disk Free: {disk.free/(1024**3):.1f}GB")
                
                # Log to file if specified
                if self.log_file:
                    with open(self.log_file, 'a') as f:
                        f.write(f"{sample['timestamp']},"
                               f"{cpu_percent},"
                               f"{memory.percent},"
                               f"{memory.used/(1024**3):.2f},"
                               f"{memory.available/(1024**3):.2f},"
                               f"{disk.free/(1024**3):.2f},"
                               f"{disk.percent}\n")
                
            except Exception as e:
                logger.warning(f"⚠️ Performance monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def stop(self):
        """Stop performance monitoring and generate report"""
        if not self.is_monitoring:
            return self.monitor_data
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        # Generate summary report
        self._generate_summary()
        return self.monitor_data
    
    def _generate_summary(self):
        """Generate performance summary"""
        if not self.monitor_data['samples']:
            return
        
        duration = datetime.now() - self.monitor_data['start_time']
        samples_df = pd.DataFrame(self.monitor_data['samples'])
        
        avg_cpu = samples_df['cpu_percent'].mean()
        avg_memory = samples_df['memory_percent'].mean()
        
        logger.info("📈 PERFORMANCE MONITORING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"   Monitoring Duration: {duration}")
        logger.info(f"   Average CPU Usage: {avg_cpu:.1f}%")
        logger.info(f"   Maximum CPU Usage: {self.monitor_data['max_cpu']:.1f}%")
        logger.info(f"   Average Memory Usage: {avg_memory:.1f}%")
        logger.info(f"   Maximum Memory Usage: {self.monitor_data['max_memory']:.1f}%")
        logger.info(f"   Samples Collected: {len(self.monitor_data['samples'])}")
        logger.info(f"   Average Memory Used: {samples_df['memory_used_gb'].mean():.1f}GB")
        logger.info(f"   Average Disk Free: {samples_df['disk_free_gb'].mean():.1f}GB")
        
        # Check for potential issues
        if self.monitor_data['max_memory'] > 90:
            logger.warning("⚠️  High memory usage detected - consider increasing memory allocation")
        if self.monitor_data['max_cpu'] < 50:
            logger.info("💡 CPU usage is low - consider increasing parallelization")
        if samples_df['disk_free_gb'].min() < 10:
            logger.warning("⚠️  Low disk space detected")