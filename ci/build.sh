#! /bin/bash
set -e

docker login -u gitlab-ci-token -p $CI_JOB_TOKEN $CI_REGISTRY

docker pull $IMAGE_TAG || docker pull $LATEST_TAG || true

docker build -f docker/Dockerfile --target prod \
    --cache-from $IMAGE_TAG --cache-from $LATEST_TAG \
    -t $IMAGE_TAG \
    .

docker push $IMAGE_TAG

