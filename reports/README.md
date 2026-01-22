# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [x] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [x] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1

> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 61 ---

### Question 2

> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s252012, s252994, s254301 ---

### Question 3

> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We used pytorch geometric, otherwise we only used frameworks/packages covered during the project. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

---
We used uv for managing dependencies, everytime we had to introduce a new dependency we used `uv add <dependency name>`. This will automatically updated the pyproject.toml aswell as the uv.lock file for exact version pinning. For a new person to get our exact development enviroment one would need to firstly install uv, secondly clone the repository and navigate to it, and finally run `uv sync` to create a virtual enviroment and install all dependencies as specified in the lock file. In order to run scripts one then would need to use `uv run`.

---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

---
We have filled out the dockerfiles, docs, models, src, and tests folder. The models folder contains only one model to test out the training locally but not for storing and versioning models. src contains all the actualy python code used in the project. We have removed only the notebooks folder as we did not use any jupyter notebooks in this project. Beyond the template we have also divided the tests folder deeper into three seperate folders: integrationtests, unittests, and performance_tests. This was done as we run some tests on every push through precommits, but the performance_tests we want to run only after a successful deployment of our service. Other than this we did not deviate from from the template.

---

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

---
Firstly, we have added docstrings to all the functions in our project. Secondly, we have also used typing in all the functions to ensure functions only run when provided with the correct input tuypes. For linting and formatting we have used ruff, in our precomit hooks we have included `ruff format --check` as well as `ruff check` to ensure we comply with formatting and it is consistent across the whole project. We do not currently employ a typechecker such as mypy in the precommit. In large projects these practices are important in order to keep the codebase easier to maintain and update, and more importantly prevent bugs.

---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

---
We have implemented four main types of tests: unit tests for the model and data, mock tests for the API endpoints, and load tests for the API. The unit tests check model input/output shapes and whether dataloaders work correctly. The API mock tests use patching to simulate services and verify that endpoints return expected responses. Load testing, performed with Locust after deployment, evaluates if the service can handle high request volumes; results are stored as CSV files in GitHub Actions artifacts.

---

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

---
The coverages for the the tests are as follows: data 16%, model 41%, and api 100%. Code coverage percentages however only include how many lines of code are ran by the tests. This does not indicate how safe our code is. For example with 100% coverage of the api there could still be errors in the case where the tests we wrote did not include every possible combinations of inputs that a user could give. And even though the coverage of data is onlly 16%, we still test all the neccesary functionality. High coverage is helpful but does not guarentee correctness

---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

---
In our project we never pushed to master and used branches for every new feature that was added. This was done to make use of github workflows which we started on pull requests such as testing and liting. Branches were not seperated among different group members as when someone struggled someone else could hop into the branch to help. When the feature the branch was meant for was finished we used pull requests to merge this into the master branch, ensuring nobody would add code directly to the master branch. We did not employ code review on pull request but in very big and critical projects this could be helpful ensuring a senior developer oversees all changes which are made to the code. However as this was a relatively small project with a lot of direct collaboration we did not think this was neccesary and would slow us down.

---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

---
We did use DVC for managing data in our project, despite the fact that the dataset is a common benchmark dataset for testing GNNs and is not being modified. For this reason DVC was not beneficial to our project. However, in a case where the data is not preprocessed and collected into a nice benchmark dataset, the main benefit of data version controll would be the ability to reproduce the experiments done on various stages of the project. It could also help with debugging specific to machine learning, as changes in the data or how it is processed can have a huge impact on the outcome.

---

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

---
We used hydra to manage all the paramaters used in our experiments. We had one central folder which stored all the config files. We supressed the outputs from hydra, as with wandb it already tracks the entire configuration that was used in training. We used a central config.yaml file with defaults to make it easy to switch between for example training:gnn or training:cnn if applicable As we used tasks with invoke to run training with a specific configuration one can simply run for example `uv run invoke train --training.epochs=40`.

---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

---
We made use of config files with Hydra, which helped us to retain information about training hyperparameters. We also used Weights and Biases to do hyperparameter sweeps, and to save our models together with data. Whenever an experiment is run, the hyperparameters are taken from a config file. Hydra automatically creates an output directory with crucial informantions. To reproduce an experiment, one would:

1. Clone the repository and run `uv sync` to get the exact dependency versions
2. Pull the data using `dvc pull` to get the correct dataset version
3. Either use the saved Hydra config from the outputs directory or download it from the W&B run page
4. Run the training with: `uv run python src/project_name/train.py --config-path /path/to/.hydra/ --config-name config`
5. Alternatively, override specific parameters: `uv run python src/project_name/train.py model.lr=0.001 model.batch_size=64`
This ensures complete reproducibility of any experiment from the project history.

---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

---
For our project we developed 3 docker images: one for training the model, and two for deployment (inference service and creating data drift report). As an example to run the docker image to run the API used for inference on port 8000 the right command is: `docker run --rm -e PORT=8000 -p 8000:8000 api:latest`. Link to the docker file: [api.dockerfile](https://github.com/KJMvanderHorst/mlops_molecules/blob/master/dockerfiles/api.dockerfile)

---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

---
The method of debugging is an individual preference, so it depended on the group member. Apart from inserting simple ```print()``` statements to check what happens inside the code, some members also used python debugger with VSCode extension. We also used the logging module (Python's built-in logger) to track important events during training, which helped identify issues in the training pipeline. We did a profiling run after the code for model training was developed.The profiling results showed that data loading and forward passes through the GNN were the main time consumers, which is expected. We didn't find any obvious inefficiencies that could be easily optimized

---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

---
We did manage to write an API for our model using FastAPI. We did this by writing a script (src/project_name/api.py) with two endpoints: `/` for checking if the service is up and `/predict` for the purpose of inference. Because the model was run in the Cloud, we also had to connect a Google Cloud bucket to store the user input and predictions, and also mount a folder from the bucket where the model was stored. Because of this we made use of `BackgroundTask` functionality from FastAPI. Apart from this we also used `lifespan()` funtion to load model only at the start of the service. We also added another script (src/project_name/drift_detection.py) for the purpose of making a report on data drifting.

---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

---
For the purposes of deployment we wrapped the model into an application using FastAPI and uvicorn and making a Docker image to pack it together. After ensuring that the service runs both locally and locally in the container, we deployed it to the cloud using Cloud Run. This was also incorporated to our workflow, so that everytime the codebase changes we ensure that the image is rebuilt and service runs with the updated code. To invoke the service an user would need to call it like that:

```bash
curl -X POST https://gcp-app-1064661101575.europe-west1.run.app/predict   -H "Content-Type: application/json"   -d '{"node_features": [[1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.7, 6.0, 9.7, 1.0, 1.0], [0.5, 0.5, 1.0, 2.0, 3.0, 4.6, 5.4, 6.2, 9.5, 1.1, 17]], "edge_index": [[0], [1]]}'
```

`node_features` must be a matrix of shape (N,11), where N is the number of atoms in the molecule, and `edge_index` must be of shape (2,n), where n represents the number of edges in the molecule.

---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

---
During this course we did manage to implement a type of monitoring which is specific to machine learning - data drifting detection. We did this by making an application using FastAPI and Evidently package. The purpose of the application is to give a report with statistics on data that is collected during inference with respect to the training data. The report can be generated by doing `curl "https://monitoring-1064661101575.europe-west1.run.app/report?n=10"`, where of course a number after n= can be changed, and corresponds to how many recent inputs should be used for the report. However, if we were given more time, we still would like to have monitoring implemented such that over time we could measure the average time it takes to recieve a user input and generate a response, and also monitor the number of requests made to the API.

---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

---
Every member of the used between 3.49 and 3.82 USD, which is around 10 USD in total throughout the whole course for the purpose of project developement. The service costing the most was Vertex AI which we used for model training, which was expected as it ran for the most ammount of time and also consumed resources like GPU which are more expensive, than for example storage. Working in the cloud was painfull, as it slowed the process of developement substantially. On the other hand, it enabled us to use previously unreachable resources such as GPU access.

---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

---
Overall the biggest struggles of the project were related to using cloud services. One of the tasks we spent most time on was pushing the docker images to the Artifact Registry using cloudbuild.yaml file. After several hours of debugging and asking a TA for help we came to the conclusion that because of the order in which we did things in the Cloud being different to what we were doing in the exercises a necessary service account was not automatically created. We fixed it simply by making a service account by hand and giving it right permissions. Another this was also making a report on data drifting, as it required connecting multiple parts of the course: Cloud deployment, interacting with buckets in the cloud and making an API that would run this. The reason for this taking so much time was that building Docker images and deploying them to the cloud simply takes a lot of time. At the beginning we also struggled with having our uv.lock file in a correct format. This was due to the fact that at first there were many thing we were doing in parallel and we were adding a lot of packages to our enviroment, which resulted in a lot of merged conflicts which were hard to resolve. Apart from those things, the rest of the tasks we accomplished went fairly smoothly.

---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
