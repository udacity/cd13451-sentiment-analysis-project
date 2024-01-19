# Sentiment Analysis with Noisy Labels using AWS SageMaker

To see if confidence learning is effective, we will apply it to a real world sentiment analysis dataset. To set the scene, suppose you are building a deep learning system for sentiment analysis of product reviews (think back to week one) for a new online beauty product store. The hope is that your system can help the company decide what to keep in stock and what to remove. The company has been tracking comments on their website and has paid annotators to provide labels. They hand you a dataset of 80,000 customer reviews, each with a label of 0 for negative sentiment or 1 for positive sentiment.

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
