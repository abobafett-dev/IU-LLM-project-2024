# Auto-Tagging of Text Using Large Language Models (LLM)

Implemented by master students of Innopolis University of the Data Science track:
* Aleksandr Vashchenko
* Grigoriy Nesterov

## 1. Summary about the project

This project is part of a project to implement a dataset tab (similar to kaggle), in this project we are implementing a tag matching recommendation system. When a user creates a new dataset and fills in tags to it, they can be generated based on the description and other information filled in for the dataset earlier.

In a current stage of project we are using the LLama-3-8b-Instruct model for tag generation and considering fine-tuning the saiga-llama3-8b model for better performance in Russian. We have a baseline of 586 tags categorized into six groups: data type, subject, geography and places, technique, task, and language. The LLM will generate tags based on these categories. When a new tag is generated, we will check its cosine similarity with the embeddings of existing tags in our database. If the similarity is high (around 1), we will use the existing tag from the database. If the similarity is low (less than 0.5), we will add the new tag to the database and store its embedding. The exact thresholds for cosine similarity will be determined later. We will select all tags that exceed a specific similarity threshold for each dataset. This process may result in some datasets having no tags or having all possible tags.

## 2. Details about the project and other

[Project Report](https://www.overleaf.com/read/qfrkdtkcgrpn#4d989a)

[Deploy Repo](https://gitlab.pg.innopolis.university/g.nesterov/tab-with-datasets-aes-ml-deploy/-/tree/main/tab_with_datasets_aes_ml_docker_compose/tags_generator?ref_type=heads)