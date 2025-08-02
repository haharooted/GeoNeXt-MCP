#!/usr/bin/env bash
# ------------------------------------------
# GeoNeXt-MCP auto-deploy w/ sslip.io | nip.io
# Ubuntu 24.04 LTS  ·  Hetzner Cloud
# ------------------------------------------
set -euo pipefail

##############  CONFIGURATION  ################
PROJECT_NAME="geonext-mcp"
APP_USER="mcp"
REPO_URL="https://github.com/haharooted/GeoNeXt-MCP.git"   # <-- edit
DNS_PROVIDER="sslip"        # sslip | nip
NAME_PREFIX=""              # optional: e.g. "api-" → api-95-217-…sslip.io
EMAIL="you@example.com"     # for certbot
GEOCODER_PROVIDER="nominatim"
NOMINATIM_URL="nominatim.openstreetmap.org"
SCHEME="https"
BING_API_KEY=""
PYTHON_VERSION="3.12"
###############################################

echo "==> Detecting public IP …"
PUBLIC_IP=$(curl -s https://ipinfo.io/ip)
DASH_IP=${PUBLIC_IP//./-}
DOMAIN="${NAME_PREFIX}${DASH_IP}.${DNS_PROVIDER}.io"
echo "    Chosen hostname: ${DOMAIN}"

echo "==> Updating OS & installing base packages …"
apt update && apt -y full-upgrade
apt install -y python${PYTHON_VERSION} python3-venv python3-pip git nginx ufw \
               certbot python3-certbot-nginx

echo "==> Creating service account & cloning repo …"
id -u "${APP_USER}" &>/dev/null || adduser --system --group --no-create-home "${APP_USER}"
install -d -o "${APP_USER}" -g "${APP_USER}" /opt/${PROJECT_NAME}
git clone "${REPO_URL}" /opt/${PROJECT_NAME}

echo "==> Python virtual-env & dependencies …"
python${PYTHON_VERSION} -m venv /opt/${PROJECT_NAME}/venv
source /opt/${PROJECT_NAME}/venv/bin/activate
pip install --upgrade pip
pip install geopy "uvicorn[standard]" fastapi mcp
[[ -f /opt/${PROJECT_NAME}/requirements.txt ]] && \
    pip install -r /opt/${PROJECT_NAME}/requirements.txt
deactivate

echo "==> systemd unit …"
cat >/etc/systemd/system/${PROJECT_NAME}.service <<SERVICE
[Unit]
Description=GeoNeXt-MCP
After=network.target
[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=/opt/${PROJECT_NAME}
Environment=GEOCODER_PROVIDER=${GEOCODER_PROVIDER}
Environment=NOMINATIM_URL=${NOMINATIM_URL}
Environment=SCHEME=${SCHEME}
Environment=BING_API_KEY=${BING_API_KEY}
ExecStart=/opt/${PROJECT_NAME}/venv/bin/python main.py --host 0.0.0.0 --port 8000
Restart=on-failure
AmbientCapabilities=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
[Install]
WantedBy=multi-user.target
SERVICE
systemctl daemon-reload && systemctl enable --now ${PROJECT_NAME}

echo "==> Firewall …"
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "==> nginx reverse proxy …"
cat >/etc/nginx/sites-available/${PROJECT_NAME}.conf <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/${PROJECT_NAME}.conf /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

echo "==> Let’s Encrypt certificate for ${DOMAIN} …"
certbot --nginx -d "${DOMAIN}" --non-interactive --agree-tos -m "${EMAIL}" --redirect

echo "✔  Done!  HTTPS endpoint: https://${DOMAIN}/"
