The code in this repo are coding notes taken while following Andrej Karpathy's recent GPT video.
The original code from Andrej Karpathy can be found at `https://github.com/karpathy/ng-video-lecture`

Dataset `shakespeare.txt` is from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

## Usage

You can run the code in this repo in a few different ways.
* ``make run``: this command will run the main function
* ``make shell``: this will create a shell inside a Docker container, 
from where you can run any code you like
* ``make notebook``: this will start a jupyter notebook server

In all cases, these commands will build the required Docker image if 
you don't already have it. 

## Development

When starting a new project, 
you first need to modify the following parameters in the ``Makefile``:
* ``IMAGE_NAME``
* ``IMAGE_TAG``

After that, you can modify the default list of the dependencies in the ``requirements.in`` file.
And when you're ready, you can build your Docker image with the ``make build`` command. 

Source code should be placed in the folder ``src``.
Unit tests should be placed in the folder ``src/tests``. 
You can run all unit tests by doing ``make tests``.

