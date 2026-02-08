## Toolstack
- Jira
- Confluence
- Github
- Colab
- VScode

## Repo structure
#### data
Since the ImagNetSubset is over 1 GB, I have added it to the gitignore, to keep the repo size small.
We will host the ImageNetSubset for training on google drive.
We can also upload the images for the final test set here.

### docs
This folder is for documentation, guides and info.


#### notebooks
In the notebooks folder we can save our colab notebooks.
I have added a colab_setup notebook that contains code to setup each notebook (git integration, paths, dependencies).

### results
Here we can put logs and results from experiments.

### scripts
The scripts folder is for small helper scripts.

#### supervised _soup
The supervised_soup folder is for our main, stable and reusable code (e.g. dataloader, model and training logic)
That way we can keep our code modular and import what we need to the notebooks.

#### Requirements and Environment
In requirements.txt we will track our dependencies to install in colab notebooks (see colab_setup notebook).
In environment.yml we track the dependencies and versions for our local environment with conda.
For the local setup see the guide.
That way we should be able to maintain reproducibility and work smoothly with the same code locally and on colab.

## Suggested workflow with git
I would suggest that whenever we work on something new, we do so on a separate branch and before we merge the other 2 review the code. For the Jira-Github integration see the guide.

## Team links
##### Github
- project repo: https://neuralspiral.github.io/deep-learning-resources/
- repo for presentations and report: https://github.com/NeuralSpiral/supervised-soup-group

##### Jira
https://stud-team-rn9zsvdn.atlassian.net/jira/software/projects/SSXP/summary

##### Confluence
https://stud-team-rn9zsvdn.atlassian.net/wiki/x/AAFI

##### Learning Resources
https://neuralspiral.github.io/deep-learning-resources/