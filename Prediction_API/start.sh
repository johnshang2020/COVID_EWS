#!/bin/bash

git clone --branch $BRANCH http://gitlab+deploy-token-2:git4fVBqxdMstTizZA9k@lvgitlabt01g1.lahey.org/clinical_machine_learning_predictions/clinical-prediction-api.git /tmp/clinical-prediction-api
# removes any previous version of the application if it is present
if test -d "/endpoint/$ENDPOINT"
then rm -rf /endpoint/$ENDPOINT
fi

cp -r /tmp/clinical-prediction-api/$ENDPOINT /endpoint/$ENDPOINT
rm -rf /tmp/clinical-prediction-api
gunicorn -b $(hostname):8000 --reload $ENDPOINT.app
exec "$@"
