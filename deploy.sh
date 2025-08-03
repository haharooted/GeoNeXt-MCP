#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GeoNeXt‑MCP one‑shot provisioning script for a fresh Ubuntu 24.04 server
# Tested on Hetzner Cloud; run as root (or with sudo -i).
# ---------------------------------------------------------------------------
set -euo pipefail

##############################################################################
# ❶ Basic variables – EDIT *ONLY* THE EMAIL ADDRESS!
##############################################################################
EMAIL="daboss@gmail.com"            # **REQUIRED** for Let's Encrypt
REPO="https://github.com/haharooted/geonext-mcp.git"
APP_DIR="/opt/geonext-mcp"
SERVICE_FILE="/etc/systemd/system/geonext-mcp.service"
NGINX_CONF="/etc/nginx/sites-available/geonext-mcp.conf"

if [[ $EUID -ne 0 ]]; then
  echo "Error: run this script as root (sudo -i)."; exit 1
fi
if [[ -z "$EMAIL" || "$EMAIL" == "your@email.address" ]]; then
  echo "Error: please set a real contact e‑mail in deploy.sh before running."; exit 1
fi

##############################################################################
# ❷ System packages
##############################################################################
echo "→ Updating apt and installing base packages …"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y -qq
apt-get install -y -qq git nginx python3 python3-venv python3-pip \
                        certbot python3-certbot-nginx curl ufw

##############################################################################
# ❸ Clone / update code and build virtual‑env
##############################################################################
echo "→ Deploying application code …"
if [[ -d "$APP_DIR/.git" ]]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO" "$APP_DIR"
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -e "$APP_DIR"

##############################################################################
# ❹ Create .env with sane defaults (skip if already present)
##############################################################################
ENV_FILE="$APP_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<EOF
# ---------------------------------------------------------------------------
# GeoNeXt‑MCP runtime options
# ---------------------------------------------------------------------------
LOG_LEVEL=info
# GEOCODER_PROVIDER=photon
# NOMINATIM_URL=nominatim.openstreetmap.org
# GEOCODER_MIN_DELAY=0
EOF
  chown root:root "$ENV_FILE"
  chmod 640 "$ENV_FILE"
fi

##############################################################################
# ❺ systemd unit
##############################################################################
echo "→ Writing systemd service …"
cat >"$SERVICE_FILE" <<'EOF'
[Unit]
Description=GeoNeXt‑MCP geocoding microservice
After=network.target

[Service]
WorkingDirectory=/opt/geonext-mcp
EnvironmentFile=/opt/geonext-mcp/.env
ExecStart=/opt/geonext-mcp/.venv/bin/python -m geonext_mcp.server
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now geonext-mcp

##############################################################################
# ❻ sslip.io hostname (▲▲ auto‑detect public IP ▲▲)
##############################################################################
IP=$(curl -s https://api.ipify.org)
DOMAIN="${IP//./-}.sslip.io"
echo "→ Using sslip.io hostname:  \e[1m$DOMAIN\e[0m"

##############################################################################
# ❼ Nginx site config – initial HTTP only (Certbot will upgrade to HTTPS)
##############################################################################
echo "→ Configuring Nginx …"
cat >"$NGINX_CONF" <<EOF
server {
    listen 80;
    server_name $DOMAIN;
    root /var/www/html;
    # Certbot will replace the block below with HTTPS config
    location / {
        proxy_pass          http://127.0.0.1:8000;
        proxy_set_header    Host \$host;
        proxy_set_header    X-Real-IP \$remote_addr;
        proxy_set_header    X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header    X-Forwarded-Proto \$scheme;

        proxy_http_version  1.1;
        proxy_set_header    Upgrade \$http_upgrade;
        proxy_set_header    Connection "upgrade";
    }
}
EOF

ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/geonext-mcp.conf
nginx -t && systemctl reload nginx

##############################################################################
# ❽ Let’s Encrypt – obtains cert *and* rewrites nginx to force HTTPS
##############################################################################
echo "→ Requesting Let’s Encrypt certificate via Certbot …"
certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email "$EMAIL" --redirect

##############################################################################
# ❾ Firewall (optional but recommended)
##############################################################################
if command -v ufw >/dev/null 2>&1; then
  echo "→ Enabling UFW firewall …"
  ufw allow 'OpenSSH'
  ufw allow 'Nginx Full'
  ufw --force enable
fi

##############################################################################
# ❿ Done 🎉
##############################################################################
echo -e "\n🚀  Deployment complete!"
echo    "    Service URL : https://$DOMAIN"
echo    "    Status      : $(systemctl is-active geonext-mcp)"
systemctl status geonext-mcp --no-pager --full
