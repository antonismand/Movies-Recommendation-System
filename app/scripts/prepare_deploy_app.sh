#!/bin/bash

set -euo pipefail

source util.sh

main() {
  # Get our working project, or exit if it's not set.
  local project_id="$(get_project_id)"
  if [[ -z "$project_id" ]]; then
    exit 1
  fi
  # Try to create an App Engine project in our selected region.
  # If it already exists, return a success ("|| true").
  echo "gcloud app create --region=$REGION"
  gcloud app create --region="$REGION" || true

  # Prepare the necessary variables for substitution in our app configuration
  # template, and create a temporary file to hold the templatized version.
  local service_name="${project_id}.appspot.com"
  local config_id=$(get_latest_config_id "$service_name")
  export TEMP_FILE="${APP}_deploy.yaml"
  < "$APP" \
    sed -E "s/SERVICE_NAME/${service_name}/g" \
    | sed -E "s/SERVICE_CONFIG_ID/${config_id}/g" \
    > "$TEMP_FILE"

  echo "To deploy:  gcloud -q app deploy $TEMP_FILE"
}

# Defaults.
APP="../app/app_template.yaml"
REGION="us-east1"
SERVICE_NAME="default"

if [[ "$#" == 0 ]]; then
  : # Use defaults.
elif [[ "$#" == 1 ]]; then
  APP="$1"
elif [[ "$#" == 2 ]]; then
  APP="$1"
  REGION="$2"
else
  echo "Wrong number of arguments specified."
  echo "Usage: deploy_app.sh [app-template] [region]"
  exit 1
fi

main "$@"