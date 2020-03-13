from __future__ import print_function, absolute_import, division

from invoke import task


@task
def dev_docker_build(ctx):
    """Build development image"""
    ctx.run('docker build -t salus-dev .', env={'DOCKER_BUILDKIT': '1'})


@task(pre=[dev_docker_build])
def dev(ctx):
    """Bring up the dev docker"""
    ctx.run('docker run -p 32771:22 --name salus-dev --rm salus-dev')


def docker_login(ctx, registry, token):
    """Login to registry"""
    ctx.run(f"docker login -u gitlab-ci-token -p {token} {registry}")


@task
def ci(ctx, registry, token, image_tag, latest_tag):
    docker_login(ctx, registry, token)
    ctx.run(f"docker pull {image_tag} || docker pull {latest_tag} || true")

    build_cmd = [
        "docker",
        "build",
        "-f", "docker/Dockerfile",
        "--target", "prod",
        "--cache-from", image_tag, "--cache-from", latest_tag,
        "-t", image_tag,
        ".",
    ]
    ctx.run(" ".join(build_cmd), env={"DOCKER_BUILDKIT": "1"})

    ctx.run(f"docker push image_tag")


@task
def ci_release(ctx, registry, token, image_tag, latest_tag):
    docker_login(ctx, registry, token)

    ctx.run(f"docker pull {image_tag}")
    ctx.run(f"docker tag {image_tag} {latest_tag}")
    ctx.run(f"docker push {latest_tag}")
