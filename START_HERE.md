# Start here - Windows submission guide

## 1. Extract
Download Taxi_Q_Learning_Assignment.zip, right-click it and choose Extract All.
Open the extracted taxi-q-learning folder. main.py should be directly inside it.
The completed report is technical_report.pdf. The notebook already contains executed outputs.

## 2. Run in VS Code (recommended before submission)
Install Python 3.11 or 3.12 if Python is not installed. In VS Code choose File > Open Folder and select taxi-q-learning. Choose Terminal > New Terminal. In a PowerShell terminal run one command at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --evaluate
```

The last command loads the saved agent and prints its test results. To train again:

```powershell
.\.venv\Scripts\python.exe main.py
```

This runs nine experiments and overwrites results. To also regenerate notebook outputs, open taxi_q_learning.ipynb in VS Code, install the Microsoft Python and Jupyter extensions if requested, select the .venv Python kernel, and click Run All. This trains again, so you do not need to run main.py first. Alternative:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

If using Git Bash instead of PowerShell, use `.venv/Scripts/python.exe` in place of `.\.venv\Scripts\python.exe`. You do not need to activate the environment.

## 3. Upload with GitHub Desktop
1. Sign into your own account in GitHub Desktop.
2. Choose File > New repository.
3. Name: taxi-q-learning. Choose a local parent folder such as Documents/GitHub. Leave Initialize README unchecked because this project provides one. Click Create repository.
4. Choose Repository > Show in Explorer to open the new repository folder.
5. Copy ALL contents from the extracted taxi-q-learning folder into this new repository folder. Copy the contents, not the outer folder. Do not copy a .venv folder if you created one; .gitignore also excludes it.
6. Return to GitHub Desktop. In the Changes tab confirm main.py, agent.py, utils.py, requirements.txt, the notebook, PDF and results files appear.
7. Enter the summary: Add Taxi Q-learning assignment. Click Commit to main (the branch name may vary).
8. Click Publish repository. To allow your teacher to view it without an invitation, uncheck Keep this code private, then click Publish repository. If the class requires a private repo, keep it private and grant the teacher access instead.
9. Choose Repository > View on GitHub. Verify the files and notebook outputs appear.
10. Copy the browser address. This is the GitHub repository link to submit, not a local folder path or a GitHub Desktop link.

## 4. Submit
Submit the repository URL and technical_report.pdf through your teacher's required submission channel. Both are required. Open the PDF to confirm the content and author name before submission. Registration number was not provided and has not been invented. If required, add it before submitting.

## Quick viva preparation
- Agent: the taxi. Environment: a simulated 5-by-5 grid with walls, passenger and destination.
- State: taxi row, taxi column, passenger location and destination, encoded as one integer.
- Action: move south/north/east/west, pick up, or drop off.
- Q-table: 500 rows by 6 columns storing estimated discounted future rewards.
- Q-learning: an off-policy, model-free temporal-difference algorithm.
- Alpha: controls the size of each learning update.
- Gamma: controls how much future rewards matter.
- Epsilon: probability of exploring a random action.
- Why decay epsilon? Explore early, use learned actions more often later.
- Why no pretrained RL package? The update and action selection are implemented directly in agent.py.
- Why can training return remain noisy? Random exploratory actions still cause penalties.
- Terminated vs truncated: successful task completion versus the 200-step time limit. Both stop the rollout, but only true termination removes the bootstrapped future value.
- main.py coordinates experiments; agent.py defines learning; utils.py handles training, evaluation, plots and GIF.
- Hyperparameter comparison: three learning rates, each trained with three random seeds. Selection uses early validation return; final test uses a separate evaluation run.
- Limitation: results apply to the small deterministic simulator, not real traffic.

Read agent.py and the report before the viva. Be able to explain the Q-learning update in your own words.
