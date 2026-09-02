FROM alpine:latest
RUN apk add --no-cache nfs-utils bash
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 2049/tcp 111/tcp 111/udp
ENTRYPOINT ["/entrypoint.sh"]
