# ── build stage ──────────────────────────────────────────────
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── runtime stage ────────────────────────────────────────────
FROM nginx:alpine

# go2rtc upstream for the /go2rtc/ proxy — the Sentinel-managed go2rtc
# container (docker-compose.yml sets the same value; override via env).
ENV GO2RTC_UPSTREAM=http://go2rtc:1984

# The stock entrypoint runs envsubst over /etc/nginx/templates/*.template
# (substituting only DEFINED env vars) into NGINX_ENVSUBST_OUTPUT_DIR — set to
# /etc/nginx so nginx.conf.template renders to the main /etc/nginx/nginx.conf.
# No static conf is copied; the template is the single source of the config.
ENV NGINX_ENVSUBST_OUTPUT_DIR=/etc/nginx
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -q --spider http://127.0.0.1/index.html || exit 1
