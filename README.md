# Sentiment Analysis with Noisy Labels using AWS SageMaker

Continuous annotation is the process of regularly updating training data with fresh examples from a currently deployed model checkpoint. These examples are then used to train the next version of the model. A significant challenge in this process is the efficient selection and annotation of new examples, especially when dealing with large, complex datasets where human labeling is prone to noise and errors.

One of the main challenges of annotation is its repetitive nature, which can lead to fatigue, coupled with the inherent ambiguity in the task itself. In our project, we aim to explore and utilize methods and libraries from cutting-edge research to tackle these challenges. Our focus is particularly on analyzing a deployed model's output to refine and enhance the training dataset.

Given that annotation errors are a common occurrence, our approach involves employing Confidence Learning. This technique uses a trained model's predictions to identify and highlight questionable labels. The premise is that instances where the model shows unexpected responses may point to potential labeling inaccuracies.

To test the effectiveness of Confidence Learning, we plan to apply it to a real-world scenario involving sentiment analysis. Imagine developing a deep learning system to analyze product review sentiments for a new online beauty product store. The system's aim is to help the company decide which products to keep and which to phase out. The company actively monitors customer feedback on its website and has engaged annotators to label these sentiments. You are provided with a dataset of 80,000 customer reviews, each tagged with a 0 for negative sentiment or a 1 for positive sentiment. This project will serve as a practical application to assess how Confidence Learning can improve the accuracy and reliability of sentiment analysis in a dynamic, real-world environment.

## Getting Started

To get a copy of the project running:

1. Proceed to open up the AWS console from the AWS Gateway.
2. Click through the AWS console to Amazon Sagemaker.
3. Click through the main Sagemaker page to "Notebook" > "Notebook instances" from the left hand toolbar, and create a notebook instance.

Then, go through the following steps:

**1. Open a Terminal in the Notebook:**
  - In the Jupyter Notebook interface, go to the "New" menu.
  - Choose "Terminal" to open a new terminal window.
    
**2. Go to SageMaker directory:**
  - `cd SageMaker`

**3. Clone the Repository:**
  - `git clone https://github.com/udacity/cd13451-sentiment-analysis-project.git`

**4. Accessing the Repository in the Notebook:**
  - After cloning, you can go back to the Jupyter interface and navigate to the cloned repository directory. You will find it in the file explorer. You then open `cd13451-sentiment-analysis-project` repository directory and navigate to the `starter` directory. Then, open `Confidence_Learning.ipynb` notebook.
    
### Dependencies
The dependencies are listed in a `requirements.txt` in 
```
starter/requirements.txt
```

## Project Instructions
The datasets for this project are provided as CSV files in the `starter/data` directory and are intended for use by the sentiment classifier. Before utilizing these datasets, you need to create embeddings for the reviews. Our training model is a 3-layer MLP that is built on top of pre-computed BERT features. This model, implemented using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html), can be found in `src/sentiment_classifier_system.py`. Rather than processing raw text, we use a pre-trained 'foundation model' to map each review text into a 768-dimensional vector embedding. The code for generating these embeddings is located in `create_embeddings.py` within the starter directory. Please follow the steps below to complete the project:

1. Explore [Cleanlab](https://docs.cleanlab.ai/v2.0.0/tutorials/indepth_overview.html#) a Python library for identifying potential label quality issues. Review the overview [here](https://docs.cleanlab.ai/v2.0.0/tutorials/indepth_overview.html#), and see an example of Cleanlab interacting with PyTorch models in an image classification tutorial [here](https://docs.cleanlab.ai/v2.0.0/tutorials/image.html).
2. Familiarize yourself with [LabelStudio](https://labelstud.io/), an open-source annotation tool.
3. Access the AWS console and select S3 from the services.
4. Within your S3 bucket (similar to `sagemaker-us-east-1-{ACCOUNT NUMBER}`), create a data folder. Confirm the bucket variable in the Confidence_Learning.ipynb notebook.
5. Download `create_embeddings.py` and the CSV files into the data folder. Create embeddings for `train.csv`, `dev.csv`, and `test.csv` files locally. Utilize a GPU for this task; you can leverage free GPUs on platforms like Google Colab, such as the T4 GPU. Upload the script and data files to Colab, then execute the following commands:

```
!python create_embeddings.py --data_path './data/train.csv' --output_path './data/train.pt'
!python create_embeddings.py --data_path './data/dev.csv' --output_path './data/dev.pt'
!python create_embeddings.py --data_path './data/test.csv' --output_path './data/test.pt'
```
Using a T4 GPU, generating `train.pt`, `dev.pt`, and `test.pt` will take approximately 10 mins, 3 mins, and 3 mins, respectively. After generating the `.pt` files, manually upload them to your S3 bucket at `s3://sagemaker-us-east-1-{ACCOUNT NUMBER}/data/`.

6. Fill out the TODO parts in the `Confidence_Learning.ipynb` and `main.py`.

## License

[License](LICENSE.txt)
