# How to install the required packages for your Jupyter notebook

1. Update `{jupyter-notebook}-requirements.txt` with the packages you need, where `{jupyter-notebook}` is the custom notebook you  want to change. If not previously done, update the corresponding `{jupyter-notebook}.Dockerfile` to use `{jupyter-notebook}-requirements.txt`.

2. Make sure you are logged in (using username and gitlab access token): 

        docker login registry.ailab.rnd.ki.sw.ericsson.se

3. Build the image:

        docker build -t registry.ailab.rnd.ki.sw.ericsson.se/fair-ai/main/fair-fl/{image_name} -f {jupyter-notebook}.Dockerfile .

    where `{image_name}` is the chosen name for the image, and `{jupyter-notebook}` is the same as in (1). Make sure to specify the platform of the base image to be `linux/amd64`. 

4. Push the image to the project container registry:

        docker push registry.ailab.rnd.ki.sw.ericsson.se/fair-ai/main/fair-fl/{image_name}
    where `{image_name}` is the image name chosen in previous step.

5. Make sure that you have added a profile, with the docker image specified, to `extra-notebook-profiles.yaml` 
    in the `getting-started/jupyter-hub` directory. Also make sure that you have added the name of an image
    pull secret to the profile. Or, you can replace the file in that directory with
    the file with the same name in this directory.

* Create the image pull secret (if not previously done):

        kubectl create secret docker-registry {name} --docker-server=registry.ailab.rnd.ki.sw.ericsson.se --docker-username={token_name} --docker-password={token}
    where `{name}` is the chosen name of the secret (`fair-fl-token` in the default file here), 
    `{token_name}` is the name of the registry pull token created for the project on gitlab, and `{token}` is
    the token itself. Or use the `create-image-pull-secret.sh` script in the `getting-started` directory.

6. Reinstall the hub using the `install-jupyter.py` script in the `getting-started/jupyter-hub` directory:

        pipenv install
        pipenv shell
        python ./install-jupyter.py uninstall fair-ai
        # Wait a couple of seconds to ensure all resources are deleted
        python ./install-jupyter.py install fair-ai