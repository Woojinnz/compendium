# Collecting data

Data collection has been done in-house using the UWB positioning system. The recorder script should be found in the positioning service folder for the UWB system.
This script replaces the normal positioning service (main.py), pushing nothing to the redis for the fall detection service.
The script records everything from the time it starts until stopped by entering "y" in the console.

Please note that as the backend is required for the UWB system to work, the backend must be running before starting the recorder script.

