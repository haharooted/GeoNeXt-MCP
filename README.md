![Showcase](/pictures/Excalidraw_GEONEXTMCP_2.png)

GeoNeXt is a MCP server built on fastmcp for enabling geolocation capability for LLMs

How to setup:
1) Setup a fresh Ubuntu 24 server (i suggest Hetzner.com) with atleast 4GB RAM
2) SSH into the server and run:
    - ``` wget https://raw.githubusercontent.com/haharooted/GeoNeXt-MCP/refs/heads/main/deploy.sh ```
    - ``` bash deploy.sh ```
3) Voilá, the MCP should be up and running - copy the IP and use it when you call OpenAI (or a self hosted model)

For adding more features, you can find docs for fastmcp here: https://gofastmcp.com/

