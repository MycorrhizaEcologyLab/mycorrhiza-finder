# MycorrhizaFinder development repository

## Note to users

This repository, and this README, are aimed at development of the source code, and not users of the MycorrhizaFinder tool. To download the tool and to see full user documentation, see https://doi.org/10.5281/zenodo.17485909.

---

## Contents

1. [Setting up your local development environment](#setting-up-your-local-development-environment)
2. [Running the tool locally](#running-the-tool-locally)
3. [Contributing to the repo](#contributing-to-the-repo)
4. [Building new versions of the tool](#building-new-versions-of-the-tool)

## Setting up your local development environment

1. If you have not done so already, [download the pre-built executables](https://doi.org/10.5281/zenodo.17485909) and follow the instructions in the quick-start user guide to set up Postgres and to check that the tool runs.

1. Install **node** from the [official website](https://nodejs.org/en/download). Version 22.13.1 is tested with the current version of the tool. For Windows, we would recommend downloading the .msi installer instead of using `fnm`. You can validate this worked by checking that the following command in terminal returns the corresponding version you have downloaded `node --version`.

2. Install **Python 3.11** from the [official website](https://www.python.org/downloads/) or from your package manager. We would recommend ticking the box asking whether you would like to add this to path. After the install, when you open a terminal and run `python --version`, the output should be `3.11.*`.

3. Check out this repo.

4. You can use any code editor you like, but the repo is optimised for [Visual Studio Code](https://code.visualstudio.com/). If you open the repo in VS Code it should automatically prompt you to install the [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and [mypy (Matan Gover)](https://marketplace.visualstudio.com/items?itemName=matangover.mypy) extensions, which should auto-run on each save. This is not essential, but if you don't use it then please do run ruff and mypy manually before committing code to the repo so we can keep it consistently formatted and typed.

5. Copy the contents of `_internal/trained_networks` from the executables into `amf-backend/trained_networks` in the repo.

6. Set the `IS_DEV` flag in `amf-backend/run_amf_api.py` to `True`. This must be set back to `False` before making production builds.

7. `cd` to `amf-backend` and run the following set of commands. (Note that line 2 is how to activate a virtualenv on Windows; if you are on MacOS or Linux, this will differ. See [here](https://docs.python.org/3/library/venv.html#how-venvs-work) for details.)

    ```
    python -m venv amfenv
    .\amfenv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r ./requirements.txt
    python -m pip install -e .
    ```

8. `cd` to `amfbrowser` react and run
    ```
    npm install
    ```

### Linux installation tips

In order to install on Linux, you will likely need to install the python dev package, libpq-dev, gcc for psycopg2 to work - you can do this with the following (note that we are using python3.11 in the below).

```
sudo apt install python3.11-dev
sudo apt install libpq-dev
sudo apt install build-essential
```

You may also need to switch your python version to 3.11 - you can use deadsnakes for this as below.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
```

Python will then be accessible using `python3.11`.


## Running the tool locally

### Running the GUI

1. In one terminal, `cd` to `amf-backend` and run
    ```
    .\amfenv\Scripts\activate
    python ./run_amf_api.py
    ```

2. In a second terminal, `cd` to `amfbrowser-react` and run
    ```
    npm start
    ```

    It will normally take a couple of minutes to start up, but eventually it should inform you that it's running successfully.

3. You can then open http://localhost:3000 and use the tool!

### Command line interface

There is also a powerful command line interface to bypass the GUI, accessible at `amf-backend/run_amf_cli.py`.

This is documented using argparse help, such that you can run
```
python run_amf_cli.py --help
```
to be informed of the available tool modes, and then, e.g. for `predict` mode,
```
python run_amf_cli.py predict --help
```
to view all the possible options.

## Contributing to the repo

The `main` branch should only be used when we want to make a public release. The `dev` branch contains the latest **reviewed and tested** changes. All work should be done in a new branch, with a pull request then submitted to merge into `dev`. Please try to use [semantic commit messages](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716).


## Building new versions of the tool

In order to release new versions of the tool, you will need to rebuild the executable. Note that for each platform you want to release the tool on, you need to build the tool on that corresponding platform. The default build is for Windows, but this can easily be adapted for MacOS and Linux.

### Generating a new executable

The main steps for generating a new executable are creating a production build of the React UI, and generating a new pyinstaller executable.

1. Ensure that you have set up your development environment following the steps above, and that the `amfenv` venv is activated.
2. Install pyinstaller if it is not already installed:
    ```
    python -m pip install pyinstaller
    ```
3. Ensure that the `IS_DEV` flag in `amf-backend/run_amf_api.py` is set to `False`.
4. Follow the steps in `build.bat` to build the executable. It is safer to copy each individual line at a time rather than running the entire shell script in one go.

    This will build an executable under the `dist` folder that you can use to run the tool. Note that this comes with an `\_internal` folder that you must provide with the exe for it to work – they must also stay in the same directory.

### Differences for MacOS and Linux

For MacOS and Linux, you must build the exe on these respective platforms. Besides this, there are two small differences:
1. You must replace the `-Recurse` flag with `-rf`;
2. In the `pyinstaller` command, replace all `;` with `:`.

## Credits

This version of the MycorrhizaFinder tool has been developed by the Royal Botanical Gardens, Kew through the Natural Capital and Ecosystem Assessment (NCEA) programme. The NCEA is Defra's largest research and development programme. It is generating a robust evidence base of the location, extent and condition of our natural capital and ecosystems, and how the state of nature is changing over time. The original tool was developed by the Sainsbury Laboratory, University of Cambridge (see Evangelisti et al. 2021, https://doi.org/10.1111/nph.17697).
