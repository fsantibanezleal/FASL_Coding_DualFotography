"""
cPanel Passenger WSGI entry point for Dash application.

Dash runs on Flask internally. Passenger imports this file
and looks for the callable named `application`.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.frontend.app import server as application
