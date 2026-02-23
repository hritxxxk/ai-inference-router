"""
Timing middleware for the AI inference router.

This module implements a FastAPI middleware that tracks and logs the processing
time for each request. It adds timing information to response headers and logs
performance metrics for monitoring and optimization purposes.
"""

import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add timing information to requests.
    
    This middleware measures the total processing time for each request,
    adds it to the response headers as X-Process-Time, and logs the
    timing information for monitoring and analysis. This is essential
    for tracking performance and identifying potential bottlenecks.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add timing information to the response.
        
        This method wraps the request processing to measure the total
        time taken from receiving the request to sending the response.
        It ensures timing is captured even if exceptions occur during
        request processing.
        
        Args:
            request: The incoming request object
            call_next: Function to call the next middleware/handler
            
        Returns:
            The response object with added timing header
        """
        start_time = time.time()
        
        response = None
        try:
            response = await call_next(request)
        finally:
            process_time = time.time() - start_time
            if response is not None:
                response.headers["X-Process-Time"] = f"{process_time:.4f}s"
            
            # Log the request timing for monitoring and analysis
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {getattr(response, 'status_code', 'error')} - "
                f"Process Time: {process_time:.4f}s"
            )
        
        return response