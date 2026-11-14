#!/usr/bin/env bash
# download_geoip.sh: fetch MaxMind GeoLite2 databases for local enrichment
#
# STATUS: unbuilt. GeoIP/ASN enrichment is on the roadmap but not implemented
#
# GeoLite2 is free but its licence does not allow redistribution, so .mmdb files cannot be committed.
# users must create their own MaxMind account and download the databases with their own licence key.
#
# Privacy note:
#   A lookup in a local .mmdb file never leaves this machine. Do NOT swap it for an online
#   IP lookup API: that would send every address you contact to a third party.
#
# TODO (geoip), once the enrichment stage exists:
#   1. read MAXMIND_ACCOUNT_ID and MAXMIND_LICENSE_KEY from the environment or a local .env
#      file (never from a CLI argument)
#   2. download GeoLite2-ASN and GeoLite2-Country from
#      https://download.maxmind.com/geoip/databases/<edition>/download?suffix=tar.gz
#      with HTTP basic auth (account id : licence key)
#   3. check the .sha256 checksum before extracting
#   4. extract only the .mmdb file into data/geoip/
#   5. print the database build date so users know when to refresh

set -euo pipefail

cat >&2 <<'EOF'
download_geoip.sh: GeoIP enrichment is not implemented yet.

When it is, you will need a free MaxMind account and licence key:
  https://www.maxmind.com/en/geolite2/signup
The downloaded .mmdb files will be in data/geoip/ and are not committed.
EOF
exit 1
