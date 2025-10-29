# Building for the Windows platform

Using `pyinstaller`, these files can be used to build a Windows executable for AMFinder. In order to build it on alternative OS', you need to use that OS and also modify the pyinstaller command as is outlined in the build.bat file.

A working Python installation is required first.

From this directory, run the following command to setup a virtual environment for the installation:

```
prepare_build.bat
```

This will take a few minutes to download and install the dependencies.

Then, run the following command to build the executable:

```
build.bat
```

The completed EXE (with `_internals` folder) will be produced into the `./dist` folder.
