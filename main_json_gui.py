"""Main entry point for JSON Manager GUI."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_gui.json_manager import json_manager_gui


if __name__ == "__main__":
    json_manager_gui.main()
