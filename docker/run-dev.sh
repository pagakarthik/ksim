xhost +local:docker
docker compose run --rm --service-ports ksim-dev bash
# tensorboard --logdir=examples/humanoid_walking_task --bind_all