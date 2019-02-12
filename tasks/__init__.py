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


@task
def build(ctx):
    ctx.run('docker build -t registry.gitlab.com/salus/salus:latest --build-arg APP_ENV=prod .',
            env={'DOCKER_BUILDKIT': '1'})

