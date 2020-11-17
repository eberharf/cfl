

How to Use Data Version Control (DVC)

# Initial Setup (I have already done this for the cfl repo)
Reference: https://dvc.org/doc/start

1. Clone your repository 
2. Install DVC: `pip install dvc`, or `conda install -c conda-forge dvc`, or follow instructions here: https://dvc.org/doc/install
3. Initialize dvc:
    - In a terminal, navigate to your git repo
    - Run `dvc init`
    - If you run `git status`, you will now see that several dvc files have been created. Git needs to keep track of these files, so run `git commit -m "Initialize DVC"`

# If DVC has already been set up for a repo
1. All you have to do locally is make sure DVC is installed (step 2 above).

# Pull Data from Remote Drive
Reference: https://dvc.org/doc/command-reference/pull

DVC associates each data file with a .dvc file. This file includes metadata about and a reference to the actual data file. Instead of tracking the data file itself, git keeps track of this .dvc file. Whenever you pull data from or push data to the remote drive, the associated .dvc file is updated to keep track of these changes. 

1. Find the corresponding .dvc file for the data you want to pull. 
2. `dvc pull /path/to/data.dvc`
3. When you do this for the first time, you will be prompted to authenticate with Google. Click on the link printed out in your terminal, copy the code that is generated at that site, and paste it back into your terminal.
4. You should now see the data file in your local directory. 

# Add New Data to Remote Drive
Reference: https://dvc.org/doc/start/data-versioning
1. First, we need to ask dvc to track the new file or directory: `dvc add /path/to/data/datafile`
2. This will generate a file called `/path/to/data/datafile.dvc` that will track the actual data file. It will also modify `/path/to/data/.gitignore` so that the actual data file is not tracked by git. Git needs to keep track of these changes: so they must be added and commited (dvc will print out the exact line you should run):
    - `git add /path/to/data/datafile.dvc /path/to/data/.gitignore`
    - `git commit -m "Add raw data"`
    - Push to your preferred git branch (i.e. `git push origin dev`)
3. The first time you do this, you need to configure remote storage so dvc knows what remote drive to push your data to (I have already done this for the cfl repo):
    - `dvc remote add -d storage gdrive://4cvftbgynmuljmknbvy/cfl_data` (replace the jumble in the middle with  the code in your google drive url)
    - `git commit .dvc/config -m "Configure remote storage"`
    - Push to your preferred git branch (i.e. `git push origin dev`)
4. Now you are ready to push your data to the remote drive: `dvc push`

# Track Changes to Your Data
1. Change your data file
2. `dvc add /path/to/data/datafile`
3. `git commit /path/to/data/datafile.dvc -m "Dataset updates"`
4. `dvc push`