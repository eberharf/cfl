

# How to Use Data Version Control (DVC)

## Initial Setup 

### (This has already been done for the cfl repo)

Reference: https://dvc.org/doc/start

1. Clone your repository 
2. Install DVC: `pip install dvc`, or `conda install -c conda-forge dvc`, or follow instructions here: https://dvc.org/doc/install
3. Initialize dvc:
    - In a terminal, navigate to your git repository
    - Run: `dvc init`
    - If you run `git status`, you will now see that several dvc files have been created. Git needs to keep track of these files, so run: `git commit -m "Initialize DVC"`

## If DVC has already been set up for a repository (i.e. by your collaborator)
1. clone your repository with git 
2. make sure DVC is installed on your local machine (see step 2 above).

## Setup remote drive for DVC storage
References: 

- https://dvc.org/doc/start/data-and-model-versioning#storing-and-sharing
- https://dvc.org/doc/command-reference/remote/add#remote-add

Note: if a collaborator has set up DVC for your repo, they may have already configured a remote storage location and you can skip this.

While git is great for software version control, git repository hosting services like GitHub often have conservative storage limits, making it a challenge to track/backup data files with git. DVC circumvents this issue by pushing data files to a remote storage device that the user specifies. This can be a lab server, Google Drive, AWS S3 bucket, or other accepted drive listed here: https://dvc.org/doc/command-reference/remote/add#supported-storage-types.

The following instructions are for how to configure a Google Drive directory as your remote storage location.

1. Navigate to drive.google.com and create a new folder where you would like to store your data with DVC for this project. I will call this folder `my_project_data`.
2. Enter the folder you just created. If you look at the website url, you will see an identification string that looks something like `4cvftbgynmuljmknbvy`. Copy this for the next step.
3. In your terminal, navigate to your git repository.
4. Configure DVC to use this remote storage location: `dvc remote add -d storage gdrive://4cvftbgynmuljmknbvy/my_project_data` (replace the jumble in the middle with the code you copied from your Google Drive url)
5. DVC will have updated your dvc config file with information about the remote storage device. Track this change with git: `git commit .dvc/config -m "Configure remote storage"`
6. Push changes: `git push origin main`
    
## Add New Data to Remote Drive
References: 
- https://dvc.org/doc/start/data-versioning
- https://dvc.org/doc/command-reference/add

Prerequisite: make sure a remote storage device has been configured with DVC for this git repository. 

1. Save your data file within your git repository locally.
2. Ask dvc to track the new file or directory: `dvc add /path/to/data/datafile`
3. This will generate a file called `/path/to/data/datafile.dvc` that will track the actual data file but doesn't contain the data itself. It will also modify `/path/to/data/.gitignore` so that the actual data file is not tracked by git. Git needs to keep track of these changes, so they must be added and commited (dvc will print out the exact line you should run):
    - `git add /path/to/data/datafile.dvc /path/to/data/.gitignore`
    - `git commit -m "Add raw data"`
    - `git push origin main`
4. Now you are ready to push your data to the remote drive: `dvc push`

## Pull Data from Remote Drive
References: 
- https://dvc.org/doc/start/data-and-model-versioning#retrieving
- https://dvc.org/doc/command-reference/pull

DVC associates each data file with a .dvc file. This file includes metadata about and a reference to the actual data file. Instead of tracking the data file itself, git keeps track of this .dvc file. Whenever you pull data from or push data to the remote drive, the associated .dvc file is updated to keep track of these changes. 

1. Find the corresponding .dvc file for the data you want to pull. 
2. `dvc pull /path/to/data.dvc`
3. When you do this for the first time, you will be prompted to authenticate with Google. Click on the link printed out in your terminal, copy the code that is generated at that site, and paste it back into your terminal.
4. You should now see the data file in your local directory. 



## Track Changes to Your Data
1. Change the contents of your data file in some way (but do not change the name of the file!) 
2. Track changes to file with DVC (this will update `datafile.dvc`): `dvc add /path/to/data/datafile`
3. Track changed `.dvc` file with git: `git commit /path/to/data/datafile.dvc -m "Dataset updates"`
4. `git push origin main`
5. Upload changed data file to remote storage: `dvc push`