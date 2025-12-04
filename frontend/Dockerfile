FROM node:20-alpine as build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install --no-audit --no-fund --legacy-peer-deps
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]


