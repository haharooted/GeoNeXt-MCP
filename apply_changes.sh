git pull
/opt/geonext-mcp/.venv/bin/pip install -e .
sudo systemctl daemon-reload
sudo systemctl restart geonext-mcp
sudo journalctl -u geonext-mcp -n 200 -f
