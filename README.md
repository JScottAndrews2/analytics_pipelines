## Introduction
This package was made as part of a presentation (Pragmatic Programming 2.0) for SIOP 2025

The following instructions will help you download the necessary programs if you want to run code live. 
If you're not familiar with this process, maybe just watch along and get the set-up after, you don't want to miss out on important info during the presentation!

For instructions on using the modules in the package, main_nb provides an example using the open-source data from the presentation.

# How to install Git
Follow the steps in the guide below...it has pictures!

Guide: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git


# How to clone the github repository
Step 1: Go to the github site here:  https://github.com/JScottAndrews2/pragmatic_programming.git
Under "Code" select clone and copy the link.

Step 2: Use the terminal to navigate to the folder where you want to put the repo clone
    e.g. cd Documents/SIOP2025

Step 3: run the following in the command line, replacing the bold with the link you copied: git clone LINK_YOU_COPIED


# How to install Anaconda 
Follow the steps in the guide below...it has pictures!

Guide: https://sparkbyexamples.com/python/install-anaconda-jupyter-notebook/
Just follow the download instructions
Download: https://www.anaconda.com/products/distribution

# How to create and Anaconda Environment

You can find the guide here: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands

Step 1: Open a command prompt by typing cmd into the search bar

You will type the following commands into the terminal and press enter.

Step 2: conda create --name pragmatic_programming python=3.9

Step 3: activate pragmatic_programming

Step 4: Navigate to the folder where you saved the repository us the cd command

    e.g. cd Documents/SIOP2025/pragmatic_programming
    
Step 5: install the required packages us the following command: pip install -r requirements.txt

# How to install PyCharm Community 
Follow the steps in the guide below...it has pictures!

Guide: https://www.guru99.com/how-to-install-python.html#:~:text=Step%201)%20To%20download%20PyCharm,setup%20wizard%20should%20have%20started.

Download: https://www.jetbrains.com/pycharm/download/#section=windows

# How to set-up a PyCharm interpretor with your Anaconda environment
Follow the steps in the guide below...it has pictures!

Guide: https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#add_new_project_interpreter

# How to upload your Anaconda environment to jupyter notebook
guide: https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084

Step 1: activate your conda environment

Step 2: type the following: conda install -c anaconda ipykernel

Step 3: type the following: python -m ipykernel install --user --name=pragmatic_programming
